import os
import ssl
import arxiv
import kuzu
import torch
import urllib.request
import streamlit as st
from typing import List, Dict, TypedDict, cast
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph, END

# Bypass SSL verification (Use only for local prototyping!)
ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "fallback_key_here")

# EDIT HERE: Change the LLM model if Groq releases a new one
LLM_MODEL = "llama-3.3-70b-versatile"

# Streamlit Cache: Connects to databases ONCE to prevent lock errors
@st.cache_resource
def initialize_databases():
    print("Initializing databases...")
    
    # 1. Vector Store (Chroma)
    embeds = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vs = Chroma(
        collection_name="research_papers",
        embedding_function=embeds,
        persist_directory="./local_chroma_db" 
    )
    
    # 2. Graph Store (Kuzu)
    k_db = kuzu.Database('./local_graph_db')
    g_conn = kuzu.Connection(k_db)
    
    return vs, g_conn

# Initialize global databases for the nodes to use
vector_store, graph_conn = initialize_databases()


# ==========================================
# 2. STATE & DATA MODELS (PYDANTIC)
# ==========================================
# Using total=False prevents Pylance/VS Code from throwing fake errors
class QueryIntent(BaseModel):
    category: str = Field(description="Classify the query into ONE of these categories: 'memory_or_casual' (greetings, chitchat, or asking about the conversation history), 'off_topic' (completely unrelated to academic research, coding, or science), or 'research' (requires searching a database for facts, papers, or technical explanations).")
class GraphState(TypedDict, total=False):
    query: str
    user_id: str
    chat_history: List[Dict[str, str]] 
    expanded_queries: List[str]
    vector_context: List[str]
    graph_context: List[str]
    merged_context: str
    used_sources: List[str] # <--- NEW: Stores the exact chunks passed to the LLM
    draft_answer: str
    eval_scores: Dict[str, float]
    iterations: int 
    external_search_done: bool 

class ExpandedQueries(BaseModel):
    queries: List[str] = Field(description="3 sub-queries to maximize search coverage.")

class QueryEntities(BaseModel):
    entities: List[str] = Field(description="Core concepts, methods, or datasets mentioned.")

class EvaluationScores(BaseModel):
    faithfulness: float = Field(description="Score 0.0 to 1.0. Is the answer fully supported by the context?")
    relevance: float = Field(description="Score 0.0 to 1.0. Does it fully answer the user's query?")

class GraphNode(BaseModel):
    id: str = Field(description="Unique identifier for the entity (e.g., 'YOLO').")
    label: str = Field(description="Type of entity. Choose from: [Concept, Method, Dataset, Metric].")

class GraphEdge(BaseModel):
    source_id: str = Field(description="The ID of the source node.")
    target_id: str = Field(description="The ID of the target node.")
    relationship: str = Field(description="How they connect. Choose from: [USES, IMPROVES, EVALUATES, DEFINES].")
    context: str = Field(description="A brief sentence explaining why they connect.")

class KnowledgeGraph(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]

class ArxivSearchQuery(BaseModel):
    search_term: str = Field(description="A 2-4 word search query optimized for ArXiv.")


# ==========================================
# 3. INGESTION & AUTONOMOUS FETCHING
# ==========================================
def setup_kuzu_schema(conn):
    try: conn.execute("CREATE NODE TABLE Entity (id STRING, label STRING, PRIMARY KEY (id))")
    except Exception: pass
    try: conn.execute("CREATE REL TABLE Connected_To (FROM Entity TO Entity, relationship STRING, context STRING)")
    except Exception: pass

def ingest_paper(file_path: str, is_public: bool, user_id: str = None):
    print(f"\n--- INGESTING: {os.path.basename(file_path)} ---")
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(pages)
    
    for chunk in chunks:
        chunk.metadata["access_level"] = "public" if is_public else "private"
        chunk.metadata["user_id"] = "system" if is_public else user_id
        chunk.metadata["source"] = os.path.basename(file_path)

    vector_store.add_documents(chunks)
    setup_kuzu_schema(graph_conn)
    llm = ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(KnowledgeGraph)
    
    # Process only first 3 chunks to save time during testing
    for chunk in chunks[:3]: 
        prompt = f"Extract a knowledge graph from this text:\n{chunk.page_content}"
        try:
            graph_data = cast(KnowledgeGraph, llm.invoke(prompt))
            for node in graph_data.nodes:
                graph_conn.execute("MERGE (n:Entity {id: $id}) ON CREATE SET n.label = $label", 
                                   parameters={"id": node.id.lower(), "label": node.label})
            for edge in graph_data.edges:
                graph_conn.execute("""
                    MATCH (source:Entity {id: $src_id}), (target:Entity {id: $tgt_id})
                    MERGE (source)-[r:Connected_To]->(target)
                    ON CREATE SET r.relationship = $rel, r.context = $ctx
                """, parameters={
                    "src_id": edge.source_id.lower(), "tgt_id": edge.target_id.lower(),
                    "rel": edge.relationship, "ctx": edge.context
                })
        except Exception: pass

def download_and_ingest(search_query: str, max_papers: int = 2):
    print(f"\n🌐 Searching arXiv for: '{search_query}'...")
    save_dir = "./downloaded_papers"
    os.makedirs(save_dir, exist_ok=True)
    
    # Polite arXiv client to avoid 429 Rate Limit bans
    client = arxiv.Client(page_size=10, delay_seconds=3.0, num_retries=5)
    search = arxiv.Search(query=search_query, max_results=max_papers, sort_by=arxiv.SortCriterion.Relevance)
    
    for paper in client.results(search):
        safe_title = "".join(x for x in paper.title if x.isalnum() or x in " -_").strip()
        file_path = os.path.join(save_dir, f"{safe_title}.pdf")
        
        if not os.path.exists(file_path):
            print(f"⬇️ Downloading: {paper.title}")
            paper.download_pdf(dirpath=save_dir, filename=f"{safe_title}.pdf")
        
        ingest_paper(file_path, is_public=True)


# ==========================================
# 4. LANGGRAPH NODES
# ==========================================
def route_initial_query(state: GraphState) -> str:
    print("\n---ROUTING INITIAL QUERY---")
    llm = ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(QueryIntent)
    
    # Give the router the chat history so it knows if "What did I just say?" is a memory question
    history_str = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in state.get("chat_history", [])])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are the routing logic for an AI. Classify the user's latest query based on the conversation history."),
        ("human", "History:\n{history}\n\nLatest Query: {query}")
    ])
    
    intent = cast(QueryIntent, (prompt | llm).invoke({"history": history_str, "query": state["query"]}))
    print(f"-> Query classified as: {intent.category.upper()}")
    
    if intent.category == "research":
        return "expand_query" # Send to the heavy RAG pipeline
    else:
        return "direct_answer" # Send to the fast, database-free node

def direct_answer_node(state: GraphState) -> GraphState:
    print("---NODE: DIRECT ANSWER (NO RAG)---")
    # This node handles greetings, memory, and rejects off-topic token-wasters
    llm = ChatGroq(model=LLM_MODEL, temperature=0.3)
    
    history_str = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in state.get("chat_history", [])])
    
    prompt = f"""You are a strict but polite academic research assistant. 
    Conversation History:
    {history_str}
    
    User Query: {state['query']}
    
    RULES:
    1. If the user is asking about the conversation history, answer them based on the history above.
    2. If the user says hello or makes casual conversation, greet them and ask what they want to research.
    3. If the user asks about ANYTHING completely unrelated to academic research, AI, coding, or science (e.g., cooking, sports, writing creative fiction), you MUST politely refuse to answer and remind them of your academic purpose.
    """
    
    response = llm.invoke(prompt).content
    return {"draft_answer": response, "used_sources": []} # Empty sources so the UI doesn't break
def expand_query_node(state: GraphState) -> GraphState:
    print("\n---NODE: EXPANDING QUERY (WITH MEMORY)---")
    llm = ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(ExpandedQueries)
    
    history_str = "".join([f"{msg['role'].capitalize()}: {msg['content']}\n" for msg in state.get("chat_history", [])])
    
    # EDIT HERE: Tweak how it expands queries
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an academic research assistant. Based on the history, expand the latest query into 3 distinct database sub-queries. Resolve pronouns."),
        ("human", "Conversation History:\n{history}\n\nLatest Query: {query}")
    ])
    chain = prompt | llm
    result = cast(ExpandedQueries, chain.invoke({"history": history_str, "query": state["query"]}))
    return {"expanded_queries": result.queries, "iterations": state.get("iterations", 0) + 1}

def retrieve_vector_node(state: GraphState) -> GraphState:
    print("---NODE: VECTOR SEARCH---")
    search_filter = {"$or": [{"access_level": {"$eq": "public"}}, {"user_id": {"$eq": state.get("user_id", "system")}}]}
    all_retrieved_docs = []
    
    for q in state.get("expanded_queries", [state["query"]]):
        docs = vector_store.similarity_search(query=q, k=5, filter=search_filter) # type: ignore
        all_retrieved_docs.extend(docs)
        
    unique_contents, final_vector_context = set(), []
    for doc in all_retrieved_docs:
        if doc.page_content not in unique_contents:
            unique_contents.add(doc.page_content)
            final_vector_context.append(f"Vector Context [Paper: {doc.metadata.get('source')}]: {doc.page_content}")
            
    return {"vector_context": final_vector_context}

def retrieve_graph_node(state: GraphState) -> GraphState:
    print("---NODE: GRAPH SEARCH---")
    llm = ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(QueryEntities)
    extracted = cast(QueryEntities, llm.invoke(f"Extract core academic entities from: {state['query']}"))
    search_entities = extracted.entities
    
    if not search_entities: return {"graph_context": []}
    
    graph_results = []
    cypher_query = """
    MATCH (source)-[r]-(neighbor)
    WHERE lower(source.id) IN $entities
    RETURN source.id, r.relationship, neighbor.id, r.context
    LIMIT 15
    """
    try:
        result = graph_conn.execute(cypher_query, parameters={"entities": [e.lower() for e in search_entities]})
        while result.has_next():
            record = result.get_next()
            graph_results.append(f"Graph: '{record[0]}' {record[1]} '{record[2]}'. Context: {record[3]}.")
    except Exception: pass
    return {"graph_context": graph_results}

def rerank_and_merge_node(state: GraphState) -> GraphState:
    print("---NODE: RERANKING (DYNAMIC THRESHOLD)---")
    all_context = state.get("vector_context", []) + state.get("graph_context", [])
    if not all_context: 
        return {"merged_context": "No relevant information found.", "used_sources": []}

    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    queries = [state["query"]] * len(all_context)
    features = tokenizer(queries, all_context, padding=True, truncation=True, return_tensors="pt")
    
    model.eval()
    with torch.no_grad():
        scores = model(**features).logits.squeeze(-1).tolist()
    if isinstance(scores, float): scores = [scores]
        
    scored_docs = sorted(list(zip(all_context, scores)), key=lambda x: x[1], reverse=True)
    
    # DYNAMIC FILTERING: Only keep documents with a positive relevance score (> 0.0)
    # This means some questions might use 2 chunks, others might use 9!
    best_docs = [doc for doc, score in scored_docs if score > 0.0][:10] 
    
    # Fallback: If nothing scored highly, just take the absolute best 1 to avoid crashing
    if not best_docs and scored_docs:
        best_docs = [scored_docs[0][0]]
        
    print(f"-> Selected {len(best_docs)} highly relevant chunks for the LLM.")
    
    return {
        "merged_context": "\n\n---\n\n".join(best_docs),
        "used_sources": best_docs # Save the clean list to the state for the UI!
    }

def generate_draft_node(state: GraphState) -> GraphState:
    print("---NODE: GENERATING ANSWER---")
    llm = ChatGroq(model=LLM_MODEL, temperature=0)
    
    # NEW STRICT PROMPT: Forces the LLM to admit ignorance if the DB is empty
    prompt = f"""You are a strict academic assistant. 
    Your ONLY source of truth is the Context below. 
    
    RULES:
    1. If the Context contains 'No relevant information found.', you MUST reply EXACTLY with: "I do not have enough information."
    2. If the Context does not explicitly contain the answer, you MUST reply EXACTLY with: "I do not have enough information."
    3. DO NOT rely on your general pre-trained knowledge. 
    4. DO NOT invent citations. Cite sources inline using the exact [Paper] tags provided.
    
    Context: {state['merged_context']}
    Query: {state['query']}"""
    
    return {"draft_answer": llm.invoke(prompt).content}

def evaluate_node(state: GraphState) -> GraphState:
    print("---NODE: EVALUATING OUTPUT---")
    llm = ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(EvaluationScores)
    prompt = f"""Evaluate this answer based on the context.
    Context: {state['merged_context']}
    Query: {state['query']}
    Answer: {state['draft_answer']}"""
    
    scores = cast(EvaluationScores, llm.invoke(prompt))
    return {"eval_scores": {"faithfulness": scores.faithfulness, "relevance": scores.relevance}}

def fetch_external_papers_node(state: GraphState) -> GraphState:
    print("\n---NODE: AUTONOMOUS EXTERNAL SEARCH---")
    llm = ChatGroq(model=LLM_MODEL, temperature=0).with_structured_output(ArxivSearchQuery)
    result = cast(ArxivSearchQuery, llm.invoke(f"Convert this query into a short search term for ArXiv: '{state['query']}'"))
    
    try: download_and_ingest(result.search_term, max_papers=2)
    except Exception as e: print(f"-> ArXiv failed: {e}")
    return {"external_search_done": True}


# ==========================================
# 5. GRAPH ROUTING & COMPILATION
# ==========================================
def route_based_on_evaluation(state: GraphState) -> str:
    print("---DECISION GATE---")
    scores = state["eval_scores"]
    if state.get("iterations", 0) > 3: return "end"
        
    if scores["faithfulness"] < 0.8: return "generate_draft"
        
    if scores["relevance"] < 0.8:
        if not state.get("external_search_done", False): return "fetch_external"
        else: return "end"
            
    return "end"

workflow = StateGraph(GraphState)

# Add all nodes (Notice we added "direct_answer" here)
workflow.add_node("direct_answer", direct_answer_node) 
workflow.add_node("expand_query", expand_query_node)
workflow.add_node("retrieve_vector", retrieve_vector_node)
workflow.add_node("retrieve_graph", retrieve_graph_node)
workflow.add_node("rerank_and_merge", rerank_and_merge_node)
workflow.add_node("generate_draft", generate_draft_node)
workflow.add_node("evaluate", evaluate_node)
workflow.add_node("fetch_external", fetch_external_papers_node)

# --- THE NEW FRONT DOOR ---
# Instead of a fixed entry point, the graph starts with our conditional router
workflow.set_conditional_entry_point(
    route_initial_query,
    {
        "expand_query": "expand_query",
        "direct_answer": "direct_answer"
    }
)

# If it went to the direct answer (fast lane), the graph ends immediately after!
workflow.add_edge("direct_answer", END)

# ... (Keep the rest of your edges exactly the same)
workflow.add_edge("expand_query", "retrieve_vector")
workflow.add_edge("expand_query", "retrieve_graph")
workflow.add_edge("retrieve_vector", "rerank_and_merge")
workflow.add_edge("retrieve_graph", "rerank_and_merge")
workflow.add_edge("rerank_and_merge", "generate_draft")
workflow.add_edge("generate_draft", "evaluate")

workflow.add_conditional_edges("evaluate", route_based_on_evaluation, {
    "generate_draft": "generate_draft", "fetch_external": "fetch_external", "end": END
})
workflow.add_edge("fetch_external", "expand_query") 

app = workflow.compile()


# ==========================================
# 6. STREAMLIT WEB UI WITH MEMORY & UPLOADS
# ==========================================
st.set_page_config(page_title="Autonomous Research Agent", page_icon="🧠", layout="wide")

# --- SIDEBAR: PDF UPLOADER ---
with st.sidebar:
    st.header("📄 Add Custom Knowledge")
    st.markdown("Upload a PDF to instantly add it to the agent's database.")
    
    uploaded_file = st.file_uploader("Upload a Research Paper (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        save_dir = "./downloaded_papers"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        with st.spinner(f"Reading, embedding, and graphing {uploaded_file.name}..."):
            try:
                ingest_paper(file_path, is_public=True)
                st.success(f"Successfully ingested {uploaded_file.name}!")
            except Exception as e:
                st.error(f"Error ingesting file: {e}")
# ... (Keep your PDF uploader code here) ...
    
    st.divider() # Adds a nice visual line
    
    # --- CLEAR MEMORY BUTTON ---
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun() # Instantly refreshes the UI
# --- MAIN UI TABS ---
st.title("🧠 Autonomous Research Agent")

# Create two tabs for the UI
tab_chat, tab_graph = st.tabs(["💬 Research Interface", "⚙️ System Architecture"])

with tab_chat:
    st.markdown("Ask me anything. If I don't know the answer, I will automatically read the latest papers from arXiv or you can upload your own PDFs in the sidebar!")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display previous conversation 
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander(f"📄 View {len(message['sources'])} Source Documents"):
                    for source in message["sources"]:
                        st.info(source)

    if user_query := st.chat_input("What would you like to research today?"):
        with st.chat_message("user"):
            st.markdown(user_query)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking... (and possibly fetching papers from arXiv)"):
                
                initial_input = {
                    "query": user_query,
                    "chat_history": st.session_state.messages[-5:], 
                    "user_id": "researcher_123", 
                    "iterations": 0,
                    "external_search_done": False
                }
                
                final_state = app.invoke(initial_input)
                final_answer = final_state.get("draft_answer", "Sorry, I couldn't generate an answer.")
                used_sources = final_state.get("used_sources", [])
                
                st.markdown(final_answer)
                
                if used_sources:
                    with st.expander(f"📄 View {len(used_sources)} Source Documents"):
                        for source in used_sources:
                            st.info(source)
                
        # Save to history
        st.session_state.messages.append({"role": "user", "content": user_query})
        st.session_state.messages.append({
            "role": "assistant", 
            "content": final_answer,
            "sources": used_sources 
        })
with tab_graph:
    st.header("LangGraph State Machine")
    st.markdown("This architecture map is **auto-generated directly from your code**. If you add or remove a node, this image will update instantly!")
    
    try:
        # Ask LangGraph to draw itself as a PNG image
        graph_image_bytes = app.get_graph().draw_mermaid_png()
        # Display the image in Streamlit
        st.image(graph_image_bytes, caption="Live Agent Architecture")
        
    except Exception as e:
        # Fallback just in case you don't have internet access (it uses mermaid.ink to render)
        st.warning("Could not render the image. Here is the raw Mermaid code:")
        st.code(app.get_graph().draw_mermaid())
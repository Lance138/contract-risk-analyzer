# Building the streamlit app for Contract Risk Analyzer

import os
import re
from pathlib import Path

import streamlit as st
from huggingface_hub import snapshot_download
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from clause_extraction import CLAUSE_TYPES_TO_CHECK, get_structured_llm, analyze_clause

# Config: model names and where the pre-built CUAD knowledge base lives
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
CUAD_HF_REPO = "Lance138/contract-risk-analyzer-vectordb" # my own Hugging Face dataset repo, holds the pre-built CUAD chroma_db
CUAD_COLLECTION_NAME = "contracts_collection"

# Page setup
st.set_page_config(page_title="Contract Risk Analyzer", page_icon="⚖️", layout="wide")
st.title("Contract Risk Analyzer")
st.caption("Upload a contract to flag risky clauses, or explore examples from a real-contract knowledge base.")

# Require an API key before anything else runs
with st.sidebar:
    api_key = st.text_input("Google API key", value=os.environ.get("GOOGLE_API_KEY", ""), type="password")

if not api_key:
    st.info("Enter your Google API key in the sidebar to get started.")
    st.stop()
os.environ["GOOGLE_API_KEY"] = api_key


# Loads the local embedding model once per session, not on every rerun
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


# Loads the Gemini chat model
@st.cache_resource(show_spinner=False)
def get_llm(_api_key):
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.1, google_api_key=_api_key)


# Downloads the pre-built CUAD vector store from Hugging Face Hub once, then connects Chroma to it
@st.cache_resource(show_spinner="Downloading contract knowledge base...")
def get_cuad_vectorstore():
    local_path = snapshot_download(repo_id=CUAD_HF_REPO, repo_type="dataset")
    return Chroma(collection_name=CUAD_COLLECTION_NAME, embedding_function=get_embeddings(), persist_directory=local_path)


# Reads an uploaded PDF or TXT file into a LangChain Document
def load_file(uploaded_file) -> Document:
    suffix = Path(uploaded_file.name).suffix.lower()
    text = "\n".join(p.extract_text() or "" for p in PdfReader(uploaded_file).pages) if suffix == ".pdf" \
        else uploaded_file.read().decode("utf-8", errors="ignore")
    return Document(page_content=text, metadata={"source": uploaded_file.name})


# Formats retrieved chunks into a labeled context string for the LLM prompt
def format_docs(docs) -> str:
    return "\n\n".join(f"[{d.metadata.get('source', d.metadata.get('title', 'unknown'))}]\n{d.page_content}" for d in docs)


# Makes an uploaded filename safe to use as a Chroma collection name
def sanitize_collection_name(name: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    name = name.strip("._-")
    if len(name) < 3:
        name = name + "_doc"
    return name[:512]

# Renders a chat interface with RAG chain, used in both tabs
def render_chat_tab(vectorstore, prompt_template: str, state_key: str, placeholder: str, source_label_key: str):
    """Shared chat UI + RAG chain, reused by both the uploaded-contract tab and the CUAD explorer tab."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = {"context": retriever | RunnableLambda(format_docs), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    if state_key not in st.session_state:
        st.session_state[state_key] = []

    for msg in st.session_state[state_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input(placeholder, key=f"{state_key}_input"):
        st.session_state[state_key].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            docs = retriever.invoke(question)
            answer = st.write_stream(chain.stream(question))
            with st.expander("Sources"):
                for d in docs:
                    st.caption(d.metadata.get(source_label_key, "unknown"))
                    st.text(d.page_content[:300])
        st.session_state[state_key].append({"role": "assistant", "content": answer})


# Runs structured clause extraction across all checked clause types for one uploaded contract
def run_full_analysis(vectorstore, api_key: str) -> list[tuple[str, "ClauseFinding"]]:
    structured_llm = get_structured_llm(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    results = []
    progress = st.progress(0.0, text="Analyzing clauses...")
    for i, clause_type in enumerate(CLAUSE_TYPES_TO_CHECK):
        finding = analyze_clause(retriever, structured_llm, clause_type, format_docs)
        results.append((clause_type, finding))
        progress.progress((i + 1) / len(CLAUSE_TYPES_TO_CHECK), text=f"Checked: {clause_type}")
    progress.empty()
    return results


# Displays the clause analysis results as a color-coded risk dashboard
def render_risk_dashboard(results):
    risk_colors = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    found = [(ct, f) for ct, f in results if f.present]
    not_found = [(ct, f) for ct, f in results if not f.present]

    if found:
        st.subheader(f"Clauses found ({len(found)})")
        for clause_type, finding in found:
            icon = risk_colors.get(finding.risk_level, "⚪")
            with st.expander(f"{icon} {clause_type} — {finding.risk_level or 'Unknown'} risk"):
                st.write(finding.explanation)
                if finding.source_text:
                    st.markdown("**Source text:**")
                    st.text(finding.source_text)

    if not_found:
        st.subheader(f"Not found ({len(not_found)})")
        st.caption(", ".join(ct for ct, _ in not_found))


# Load cached resources once
embeddings = get_embeddings()
llm = get_llm(api_key)
cuad_vectorstore = get_cuad_vectorstore()
st.sidebar.metric("Reference contracts loaded", len(cuad_vectorstore.get(include=[])["ids"]))

tab_analyze, tab_explore = st.tabs(["📄 Analyze a contract", "🔎 Explore reference contracts"])

# Tab 1: upload and analyze the user's own contract
with tab_analyze:
    uploaded_file = st.file_uploader("Upload a contract (PDF or TXT)", type=["pdf", "txt"])

    if uploaded_file:
        # Only re-chunk/re-embed if this is a new file, not on every rerun
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            with st.spinner("Chunking and embedding your contract..."):
                splitter = RecursiveCharacterTextSplitter(chunk_size=1800, chunk_overlap=250, separators=["\n\n", "\n", ". ", " ", ""])
                chunks = splitter.split_documents([load_file(uploaded_file)])
                session_store = Chroma(
                    collection_name=sanitize_collection_name(f"session_{uploaded_file.name}"),
                    embedding_function=embeddings,
                )
                session_store.add_documents(chunks)
                st.session_state.session_vectorstore = session_store
                st.session_state.uploaded_filename = uploaded_file.name
                st.session_state.contract_messages = []
                st.session_state.pop("analysis_results", None)  # clear any prior file's results
            st.success(f"Indexed {len(chunks)} chunk(s) from {uploaded_file.name}.")

        # Trigger and display the structured risk analysis
        if st.button("🔍 Run risk analysis", type="primary"):
            with st.spinner("Analyzing contract for risky clauses..."):
                st.session_state.analysis_results = run_full_analysis(st.session_state.session_vectorstore, api_key)

        if "analysis_results" in st.session_state:
            render_risk_dashboard(st.session_state.analysis_results)

        st.divider()

        # Reactive Q&A chat, in addition to the proactive analysis above
        render_chat_tab(
            st.session_state.session_vectorstore,
            "Answer based only on the following context from the uploaded contract. "
            "If the answer isn't in the context, say so clearly.\n\n{context}\n\nQuestion: {question}",
            "contract_messages",
            "Ask about this contract (e.g. 'Is there an auto-renewal clause?')",
            "source",
        )
    else:
        st.info("Upload a contract above to get started.")

# Tab 2: explore the CUAD reference knowledge base
with tab_explore:
    st.caption("Ask about how specific clause types typically appear across 408 real commercial contracts.")
    render_chat_tab(
        cuad_vectorstore,
        "Answer using examples from the following excerpts of real contracts. Cite which contract each comes from.\n\n{context}\n\nQuestion: {question}",
        "cuad_messages",
        "e.g. 'Show me examples of auto-renewal clauses'",
        "title",
    )
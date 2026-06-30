"""
Private Multi-PDF RAG Chatbot
------------------------------
- Parse multiple PDFs using LlamaParse
- View parsed text and chunks
- Local embeddings (SentenceTransformers) + FAISS vector search
- Answer generation using Groq (Llama models)
- Greeting detection for natural small talk
- Everything in a single file: app.py

Run:
    pip install streamlit groq llama-parse faiss-cpu sentence-transformers langchain python-dotenv numpy
    streamlit run app.py

.env file required (same folder as app.py):
    GROQ_API_KEY=your_groq_api_key_here
    LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here
"""

import os
import re
import tempfile
import numpy as np
import faiss
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv

from groq import Groq
from llama_parse import LlamaParse
from sentence_transformers import SentenceTransformer

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False

# ============================================================
# ENV / CONFIG
# ============================================================
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
]

# ============================================================
# PAGE CONFIG & STYLE
# ============================================================
st.set_page_config(page_title="Private PDF RAG Chatbot", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
    .app-header {
        background: linear-gradient(135deg, #065f46 0%, #064e3b 100%);
        padding: 2rem; border-radius: 12px; color: white; margin-bottom: 1.5rem;
    }
    .app-header h1 { margin: 0; }
    .app-header p { margin: 0.3rem 0 0 0; opacity: 0.9; }
    .chunk-card {
        background: #f0fdf4; border-left: 5px solid #10b981;
        padding: 1rem; margin-bottom: 0.75rem; border-radius: 4px; color: #064e3b;
    }
    .source-pill {
        display: inline-block; background: #ecfdf5; color: #047857;
        border: 1px solid #a7f3d0; padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; margin-right: 6px;
    }
    .stChatMessage { border-radius: 15px; }
</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="app-header"><h1>🛡️ Private PDF Intelligence</h1>'
    '<p>Multi-PDF parsing (LlamaParse) + Local Embeddings (FAISS) + Groq Inference</p></div>',
    unsafe_allow_html=True,
)

# ============================================================
# SESSION STATE
# ============================================================
DEFAULT_STATE = {
    "chunks": [],            # [{"text":..., "source":...}]
    "faiss_index": None,
    "chat_history": [],      # [{"role": "user"/"assistant", "content": ...}]
    "files_processed": [],   # list of filenames already indexed
    "parsed_docs": {},       # {filename: full_text} for the "Parsed Text" view
    "last_retrieved": [],    # last retrieved chunks, for debug view
}
for key, val in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ============================================================
# CACHED RESOURCES
# ============================================================
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    # Runs fully locally on CPU. No API key needed.
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_groq_client():
    if not GROQ_API_KEY:
        return None
    return Groq(api_key=GROQ_API_KEY)


embed_model = load_embedding_model()

# ============================================================
# GREETING DETECTION
# ============================================================
GREETING_PATTERNS = [
    r"^\s*(hi|hello|hey|namaste|namaskar|yo|hola)\s*[!.]*\s*$",
    r"^\s*(good\s*(morning|afternoon|evening))\s*[!.]*\s*$",
    r"^\s*(how are you|k cha|k xa|sanchai)\s*[?]*\s*$",
]

GREETING_REPLY = (
    "Hello! 👋 I'm your PDF assistant. Upload one or more PDFs in the "
    "**Upload** tab and click *Index Documents*, then come back here and "
    "ask me anything about their content."
)


def is_greeting(text: str) -> bool:
    text = text.strip().lower()
    return any(re.match(p, text) for p in GREETING_PATTERNS)


# ============================================================
# CORE FUNCTIONS
# ============================================================
def parse_pdf_file(file_bytes: bytes, filename: str) -> str:
    """Parse a single PDF (as raw bytes) using LlamaParse and return its text."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / filename
        with open(tmp_path, "wb") as f:
            f.write(file_bytes)

        parser = LlamaParse(api_key=LLAMA_API_KEY, result_type="markdown")
        documents = parser.load_data(str(tmp_path))
        return "\n\n".join(d.text for d in documents)


def get_chunks(text: str, filename: str, size: int = 1000, overlap: int = 150):
    """Split text into overlapping chunks, tagged with the source filename."""
    overlap = min(overlap, size - 1)  # guard against overlap >= size (infinite loop bug)

    if HAS_LANGCHAIN:
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        texts = splitter.split_text(text)
    else:
        step = max(size - overlap, 1)
        texts = [text[i:i + size] for i in range(0, len(text), step)]

    texts = [t.strip() for t in texts if t.strip()]
    return [{"text": t, "source": filename} for t in texts]


def build_index():
    """(Re)build the FAISS index from all chunks currently in session state."""
    if not st.session_state.chunks:
        st.session_state.faiss_index = None
        return

    texts = [c["text"] for c in st.session_state.chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True)
    embeddings = np.asarray(embeddings, dtype="float32")
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    st.session_state.faiss_index = index


def retrieve_context(query: str, top_k: int = 4):
    """Retrieve the top_k most relevant chunks for a query."""
    if st.session_state.faiss_index is None or not st.session_state.chunks:
        return []

    q_emb = embed_model.encode([query], convert_to_numpy=True)
    q_emb = np.asarray(q_emb, dtype="float32")
    faiss.normalize_L2(q_emb)

    top_k = min(top_k, len(st.session_state.chunks))  # guard: top_k > number of chunks
    scores, idxs = st.session_state.faiss_index.search(q_emb, top_k)

    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx == -1:
            continue
        results.append({**st.session_state.chunks[idx], "score": float(score)})
    return results


def ask_groq(query: str, contexts: list, model: str) -> str:
    client = load_groq_client()
    if client is None:
        return "⚠️ GROQ_API_KEY not set. Please add it to your .env file."

    if not contexts:
        return "I couldn't find anything relevant to that question in the uploaded PDFs."

    context_text = "\n\n---\n\n".join(
        f"Source: {c['source']}\n{c['text']}" for c in contexts
    )

    sys_prompt = (
        "You are a helpful assistant. Answer the user's question using ONLY the "
        "provided context. Always mention which source filename the answer came "
        "from. If the answer isn't in the context, say you don't know rather than "
        "guessing."
    )
    user_content = f"Context:\n{context_text}\n\nQuestion: {query}"

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"⚠️ Groq API error: {e}"


# ============================================================
# SIDEBAR — SETTINGS
# ============================================================
with st.sidebar:
    st.title("⚙️ Settings")

    if not GROQ_API_KEY:
        st.error("GROQ_API_KEY not found in .env")
    if not LLAMA_API_KEY:
        st.error("LLAMA_CLOUD_API_KEY not found in .env")

    selected_model = st.selectbox("Groq Model", GROQ_MODELS)
    chunk_size = st.slider("Chunk Size", 300, 2000, 1000, step=100)
    chunk_overlap = st.slider("Chunk Overlap", 0, 500, 150, step=50)
    top_k = st.slider("Context Chunks (top_k)", 1, 10, 4)

    st.divider()
    if st.session_state.files_processed:
        st.markdown("**📚 Indexed files:**")
        for fname in st.session_state.files_processed:
            st.markdown(f"- {fname}")

    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# ============================================================
# TABS
# ============================================================
tab_upload, tab_chat, tab_parsed, tab_chunks, tab_debug = st.tabs(
    ["📤 Upload", "💬 Chat", "📄 Parsed Text", "🧩 Chunks", "🔍 Retrieved (debug)"]
)

# ---- TAB: UPLOAD ----
with tab_upload:
    st.subheader("Upload and index your PDFs")
    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)

    if st.button("🚀 Index Documents", type="primary"):
        if not GROQ_API_KEY or not LLAMA_API_KEY:
            st.error("Missing API keys in .env file! Add GROQ_API_KEY and LLAMA_CLOUD_API_KEY.")
        elif not files:
            st.warning("Please upload at least one PDF first.")
        else:
            progress = st.progress(0.0, text="Starting...")
            new_files = [f for f in files if f.name not in st.session_state.files_processed]

            if not new_files:
                st.info("All uploaded files are already indexed.")
            else:
                for i, f in enumerate(new_files):
                    progress.progress(i / len(new_files), text=f"Parsing {f.name}...")
                    try:
                        text = parse_pdf_file(f.read(), f.name)
                        st.session_state.parsed_docs[f.name] = text

                        new_chunks = get_chunks(text, f.name, size=chunk_size, overlap=chunk_overlap)
                        st.session_state.chunks.extend(new_chunks)
                        st.session_state.files_processed.append(f.name)
                    except Exception as e:
                        st.error(f"Failed to parse {f.name}: {e}")

                progress.progress(0.9, text="Building FAISS index...")
                build_index()
                progress.progress(1.0, text="Done!")
                st.success(f"Successfully indexed {len(st.session_state.files_processed)} file(s), "
                           f"{len(st.session_state.chunks)} total chunks.")

# ---- TAB: CHAT ----
with tab_chat:
    if not st.session_state.files_processed:
        st.info("👈 Upload and index a PDF in the Upload tab to start chatting.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Ask about your documents, or just say hi 👋")

    if query:
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            if is_greeting(query):
                answer = GREETING_REPLY
                st.markdown(answer)
            elif st.session_state.faiss_index is None:
                answer = "I don't have any PDFs indexed yet. Please upload and index a PDF first."
                st.warning(answer)
            else:
                with st.spinner("Searching documents and generating answer..."):
                    context = retrieve_context(query, top_k=top_k)
                    st.session_state.last_retrieved = context
                    answer = ask_groq(query, context, selected_model)

                st.markdown(answer)

                if context:
                    sources = sorted(set(c["source"] for c in context))
                    st.markdown(
                        " ".join(f'<span class="source-pill">📎 {s}</span>' for s in sources),
                        unsafe_allow_html=True,
                    )
                    with st.expander("Show retrieved chunks"):
                        for c in context:
                            st.markdown(
                                f'<div class="chunk-card"><b>{c["source"]}</b> '
                                f'(score: {c["score"]:.3f})<br>{c["text"][:400]}...</div>',
                                unsafe_allow_html=True,
                            )

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

# ---- TAB: PARSED TEXT ----
with tab_parsed:
    st.subheader("Full parsed text per file")
    if not st.session_state.parsed_docs:
        st.info("No parsed PDFs yet. Go to the Upload tab first.")
    else:
        selected_file = st.selectbox("Select a file", list(st.session_state.parsed_docs.keys()))
        st.text_area("Parsed text", st.session_state.parsed_docs[selected_file], height=500)

# ---- TAB: CHUNKS ----
with tab_chunks:
    st.subheader("All chunks created from your PDFs")
    if not st.session_state.chunks:
        st.info("No chunks yet. Go to the Upload tab first.")
    else:
        filenames = sorted(set(c["source"] for c in st.session_state.chunks))
        filter_file = st.selectbox("Filter by file", ["All"] + filenames)

        chunks_to_show = st.session_state.chunks
        if filter_file != "All":
            chunks_to_show = [c for c in chunks_to_show if c["source"] == filter_file]

        st.caption(f"Showing {len(chunks_to_show)} chunk(s)")
        for i, c in enumerate(chunks_to_show):
            st.markdown(
                f'<div class="chunk-card"><b>{c["source"]}</b> — chunk #{i}<br>{c["text"]}</div>',
                unsafe_allow_html=True,
            )

# ---- TAB: RETRIEVED (debug) ----
with tab_debug:
    st.subheader("Chunks retrieved for the last question")
    if not st.session_state.last_retrieved:
        st.info("Ask a question in the Chat tab to see retrieved chunks here.")
    else:
        for c in st.session_state.last_retrieved:
            st.markdown(
                f'<div class="chunk-card"><b>{c["source"]}</b> '
                f'(similarity: {c["score"]:.3f})<br>{c["text"]}</div>',
                unsafe_allow_html=True,
            )
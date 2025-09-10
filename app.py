# app.py
import os
import io
from typing import List, Tuple

import streamlit as st

try:
    import pdfplumber
    from PIL import Image
    import pytesseract
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import InferenceApi
except ImportError as e:
    st.error(f"Missing dependency: {e}. Please install all required packages.")
    st.stop()

# -----------------------
# Config / Globals
# -----------------------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HF_MODELS = {
    "granite-3.3-2b-instruct": "ibm-granite/granite-3.3-2b-instruct",
    "granite-3.3-8b-instruct": "ibm-granite/granite-3.3-8b-instruct",
    "granite-7b-base (non-instruct)": "ibm-granite/granite-7b-base"
}

# -----------------------
# Utilities
# -----------------------
@st.cache_resource
def load_embedder(model_name=EMBED_MODEL_NAME):
    return SentenceTransformer(model_name)

def pdf_to_pages_bytes(pdf_bytes: bytes) -> List[Tuple[int, str]]:
    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            txt = page.extract_text() or ""
            if txt.strip() == "":
                try:
                    pil = page.to_image(resolution=300).original
                    txt = pytesseract.image_to_string(pil)
                except Exception:
                    txt = ""
            pages_text.append((i+1, txt))
    return pages_text

def chunk_text(text: str, max_chars=1000, overlap=200) -> List[str]:
    chunks = []
    start = 0
    L = len(text)
    if L == 0:
        return []
    while start < L:
        end = min(L, start + max_chars)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(0, end - overlap)
    return chunks

def build_index(embeddings: np.ndarray):
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    return index

def normalize(vecs: np.ndarray):
    faiss.normalize_L2(vecs)
    return vecs

# -----------------------
# Hugging Face Inference wrapper
# -----------------------
class HFGenerator:
    def __init__(self, model_id: str, hf_token: str):
        self.model_id = model_id
        self.hf_token = hf_token
        self.client = InferenceApi(repo_id=model_id, token=hf_token)

    def generate(self, prompt: str, max_new_tokens: int = 512, temperature: float = 0.0):
        try:
            response = self.client(
                {
                    "inputs": prompt,
                    "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
                    "options": {"use_cache": True, "wait_for_model": True},
                },
                headers={"Accept": "application/json"},
            )
            if isinstance(response, list):
                text = response[0].get("generated_text") or response[0].get("text") or str(response)
            elif isinstance(response, dict):
                text = response.get("generated_text") or response.get("text") or str(response)
            else:
                text = str(response)
            return text
        except Exception as e:
            return f"Error calling Hugging Face Inference API: {e}"

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="📚 StudyMate — Granite (Hugging Face) + RAG", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4F8BF9;'>📖 StudyMate — PDF Q&A (IBM Granite on Hugging Face)</h1>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Settings")
hf_token = st.sidebar.text_input("🔑 Hugging Face API Token", value=os.getenv("HUGGINGFACE_API_TOKEN",""), type="password")
model_choice = st.sidebar.selectbox("🤖 Granite model (Hugging Face repo)", list(DEFAULT_HF_MODELS.keys()))
model_id = DEFAULT_HF_MODELS[model_choice]

max_context_chunks = st.sidebar.slider("🧩 Context chunks to retrieve (k)", min_value=1, max_value=8, value=3)
chunk_max_chars = st.sidebar.number_input("✂️ Chunk size (chars)", min_value=200, max_value=5000, value=1200, step=100)
chunk_overlap = st.sidebar.number_input("🔄 Chunk overlap (chars)", min_value=0, max_value=2000, value=200, step=50)
top_k_sources = st.sidebar.slider("🔝 Show top source chunks", 1, 10, 3)

if hf_token.strip() == "":
    st.sidebar.warning("⚠️ Enter your Hugging Face API token to enable generation.")

uploaded = st.file_uploader("📤 Upload PDF(s) (you can upload multiple)", type=["pdf"], accept_multiple_files=True)

if 'corpus' not in st.session_state:
    st.session_state.corpus = []

embedder = load_embedder()

if uploaded:
    with st.spinner("⏳ Processing uploaded PDFs ..."):
        processed_any = False
        progress = st.progress(0)
        total_files = len(uploaded)
        for idx_up, up in enumerate(uploaded):
            try:
                pdf_bytes = up.read()
                if not pdf_bytes or len(pdf_bytes) == 0:
                    st.error(f"❌ {up.name} is empty or could not be read.")
                    continue
                try:
                    pages = pdf_to_pages_bytes(pdf_bytes)
                except Exception as e:
                    st.error(f"❌ Error opening {up.name}: {e}")
                    continue
                if not pages or len(pages) == 0:
                    st.warning(f"⚠️ No readable pages found in {up.name}.")
                    continue
                valid_page_found = False
                for page_num, text in pages:
                    if not text or len(text.strip()) == 0:
                        st.warning(f"⚠️ Page {page_num} in {up.name} is empty or unreadable.")
                        continue
                    chunks = chunk_text(text, max_chars=chunk_max_chars, overlap=chunk_overlap)
                    if not chunks:
                        st.warning(f"⚠️ No text chunks extracted from page {page_num} in {up.name}.")
                        continue
                    for c in chunks:
                        st.session_state.corpus.append({"source_file": up.name, "page": page_num, "text": c})
                        valid_page_found = True
                if valid_page_found:
                    processed_any = True
            except Exception as e:
                st.error(f"❌ Unexpected error processing {up.name}: {e}")
            progress.progress((idx_up+1)/total_files)
        progress.empty()
    if processed_any:
        st.success(f"✅ Added content. Total chunks in corpus: {len(st.session_state.corpus)}")
        # Always rebuild index after new content
        with st.spinner("🔎 Embedding chunks ..."):
            texts = [c["text"] for c in st.session_state.corpus]
            embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
            normalize(embeddings)
            index = build_index(embeddings)
            st.session_state.index = index
            st.session_state.embeddings = embeddings
            st.session_state.texts = texts
        st.success("✅ Index built and cached in session.")
    else:
        st.error("❌ No valid content was extracted from the uploaded PDFs. Please check your files and try again.")

# Ensure session state variables are always initialized
if 'index' not in st.session_state:
    st.session_state.index = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'texts' not in st.session_state:
    st.session_state.texts = []

st.markdown("---")
st.markdown("<h2 style='color: #4F8BF9;'>💬 Ask a question (uses RAG + Granite model)</h2>", unsafe_allow_html=True)

question = st.text_area("📝 Your question", height=120)
if st.button("🚀 Answer"):
    if hf_token.strip() == "":
        st.error("❌ Please provide Hugging Face API token in the sidebar.")
    elif question.strip() == "":
        st.warning("⚠️ Please write a question.")
    elif len(st.session_state.corpus) == 0:
        st.error("❌ No uploaded PDFs / corpus available.")
    else:
        # Build index if missing
        if not st.session_state.index or not st.session_state.texts:
            with st.spinner("🔎 Building index from corpus ..."):
                texts = [c["text"] for c in st.session_state.corpus]
                embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=True)
                normalize(embeddings)
                index = build_index(embeddings)
                st.session_state.index = index
                st.session_state.embeddings = embeddings
                st.session_state.texts = texts
        if not st.session_state.index or not st.session_state.texts:
            st.error("❌ Index build failed. Please check your PDFs and try again.")
        else:
            q_emb = embedder.encode([question], convert_to_numpy=True)
            normalize(q_emb)
            D, I = st.session_state.index.search(q_emb, max_context_chunks)
            top_idxs = I[0].tolist()
            retrieved_texts = []
            context_blocks = []
            for idx in top_idxs:
                chunk_text = st.session_state.texts[idx]
                meta = st.session_state.corpus[idx]
                context_blocks.append(f"Source: {meta.get('source_file','?')} page {meta.get('page','?')}\n{chunk_text}")
                retrieved_texts.append(chunk_text)

            prompt = (
                "You are a helpful tutor. Use the following extracted passages from a student's PDFs to answer the question.\n\n"
                "Context passages:\n"
            )
            for i, cb in enumerate(context_blocks, start=1):
                prompt += f"[Passage {i}]\n{cb}\n\n"
            prompt += (
                "When answering, explicitly cite which passage(s) you used (e.g., Passage 1). "
                "If the answer is not in the passages, say so and do not hallucinate.\n\n"
                f"Question: {question}\n\nAnswer:"
            )

            st.write("### 🎯 Retrieved context (for transparency)")
            for i, cb in enumerate(context_blocks, start=1):
                st.write(f"**Passage {i} — {st.session_state.corpus[top_idxs[i-1]]['source_file']} (page {st.session_state.corpus[top_idxs[i-1]]['page']})**")
                st.write(cb[:800] + ("..." if len(cb) > 800 else ""))

            st.write("### 🤖 Model answer")
            with st.spinner("🧠 Calling Hugging Face Inference API (Granite model)..."):
                generator = HFGenerator(model_id=model_id, hf_token=hf_token)
                raw = generator.generate(prompt, max_new_tokens=400, temperature=0.0)
                st.markdown(raw)

            st.write("### 📚 Top source chunks (original)")
            for rank, idx in enumerate(top_idxs[:top_k_sources], start=1):
                if idx < len(st.session_state.corpus) and idx < len(st.session_state.texts):
                    meta = st.session_state.corpus[idx]
                    st.markdown(f"**{rank}. {meta.get('source_file')} — page {meta.get('page')}**")
                    st.write(st.session_state.texts[idx])
                else:
                    st.warning(f"Chunk index {idx} is out of range.")

st.markdown("---")
st.write("**Notes & troubleshooting**")
st.markdown(
    "- Use a smaller Granite model (eg. `granite-3.3-2b-instruct`) for quick tests; larger models require more resources and may take longer. "
    "- The Hugging Face Inference API may impose rate limits/costs depending on your account. "
)

st.write("**Model references**: IBM Granite family on Hugging Face / IBM docs.")
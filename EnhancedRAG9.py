# ENHANCEDRAG9 (Dense + Sparse Prefilter + Eval Comparison).py

# Combined RAG app + evaluation additions (Recall@K, Precision@K, EM, F1, ROUGE-1, ROUGE-L, BERTScore fallback)
# + UI controls: top_m_for_cross, answer display mode, retrieval times saved in eval.
# + Sparse-first retrieval with BM25 prefilter -> dense -> cross-encoder.
# + NEW: Evaluation toggle to compare dense-only vs sparse+dense retrieval.
"""
pip install chromadb
pip install streamlit
pip install -U langchain-groq==0.3.8
pip install -U langchain-community==0.3.31
pip install langchain-core==0.3.79
pip install sentence-transformers==5.1.2
pip install transformers==4.57.1
pip install pymupdf
pip install pypdf
pip install matplotlib"""
import os
import re
import time
import json
import uuid
import hashlib
import importlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import chromadb

from sentence_transformers import SentenceTransformer
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

# LLM wrapper (Groq)
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    PyPDFLoader = None
    RecursiveCharacterTextSplitter = None
    ChatGroq = None
    HumanMessage = None
    LANGCHAIN_AVAILABLE = False

from sklearn.metrics.pairwise import cosine_similarity

# optional tiktoken
try:
    tiktoken = importlib.import_module("tiktoken")
except Exception:
    tiktoken = None

# optional BERTScore
try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except Exception:
    bert_score_fn = None
    BERTSCORE_AVAILABLE = False

# plotting fallback
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = None
    MATPLOTLIB_AVAILABLE = False

# optional BM25 sparse retrieval
try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    BM25_AVAILABLE = False

# ------------------------
# Basic utils & metrics
# ------------------------
MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)

def sha256_text(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
def simple_sent_tokenize(text: str) -> List[str]:
    sents = [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]
    if not sents:
        return [text.strip()] if (text or "").strip() else []
    return sents

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_simple(s: str) -> List[str]:
    return re.findall(r"\w+", (s or "").lower())

def compute_exact_and_f1(pred: str, gold: str) -> Tuple[int, float]:
    em = 1 if normalize_text(pred) == normalize_text(gold) else 0
    p_tokens = tokenize_simple(pred)
    g_tokens = tokenize_simple(gold)
    if len(p_tokens) == 0 and len(g_tokens) == 0:
        return em, 1.0
    if len(p_tokens) == 0 or len(g_tokens) == 0:
        return em, 0.0
    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    if match == 0:
        return em, 0.0
    prec = match / len(p_tokens)
    rec = match / len(g_tokens)
    f1 = 2 * prec * rec / (prec + rec)
    return em, f1

def token_precision(pred: str, gold: str) -> float:
    p_tokens = tokenize_simple(pred)
    g_tokens = tokenize_simple(gold)
    if len(p_tokens) == 0:
        return 0.0
    common = {}
    for t in p_tokens:
        common[t] = common.get(t, 0) + 1
    match = 0
    for t in g_tokens:
        if common.get(t, 0) > 0:
            match += 1
            common[t] -= 1
    return match / max(1, len(p_tokens))

# ---------- ROUGE-1 (unigram F1) ----------
import collections
def rouge_n_f1(pred: str, gold: str, n: int = 1) -> float:
    def ngrams(tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    p_tokens = tokenize_simple(pred)
    g_tokens = tokenize_simple(gold)
    if not p_tokens or not g_tokens:
        return 0.0
    p_ngrams = ngrams(p_tokens, n)
    g_ngrams = ngrams(g_tokens, n)
    if not p_ngrams or not g_ngrams:
        return 0.0
    p_counts = collections.Counter(p_ngrams)
    g_counts = collections.Counter(g_ngrams)
    match = 0
    for ng, cnt in p_counts.items():
        match += min(cnt, g_counts.get(ng, 0))
    prec = match / max(1, sum(p_counts.values()))
    rec = match / max(1, sum(g_counts.values()))
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def rouge_1(pred: str, gold: str) -> float:
    return rouge_n_f1(pred, gold, n=1)

# ---------- ROUGE-L (LCS-based) ----------
def _lcs_length(a: List[str], b: List[str]) -> int:
    if not a or not b:
        return 0
    n, m = len(a), len(b)
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        curr = [0] * (m + 1)
        ai = a[i - 1]
        for j in range(1, m + 1):
            if ai == b[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = prev[j] if prev[j] >= curr[j - 1] else curr[j - 1]
        prev = curr
    return prev[m]

def rouge_l(pred: str, gold: str) -> float:
    p = tokenize_simple(pred)
    g = tokenize_simple(gold)
    if not p or not g:
        return 0.0
    lcs = _lcs_length(p, g)
    prec = lcs / len(p)
    rec = lcs / len(g)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

# ---------- Provenance stripping ----------
_PROV_SPLIT_RE = re.compile(r"\n\s*(\[\d+\]\s*Source:|→\s*Source:|Source:|Source\s*[:\-])", flags=re.IGNORECASE)
def strip_provenance(text: str) -> str:
    if not text:
        return ""
    parts = _PROV_SPLIT_RE.split(text)
    cleaned = parts[0] if parts else text
    cleaned = re.sub(r"\(chunk:?[^\)]*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\[\(]\s*\d+\s*[\]\)]", "", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned).strip()
    return cleaned

# ------------------------
# Plot helpers
# ------------------------
def plot_hist(series: pd.Series, title: str, bins: int = 20):
    arr = None
    try:
        arr = np.array(series.dropna().astype(float))
    except Exception:
        try:
            arr = np.array([x for x in series if x is not None])
        except Exception:
            arr = np.array([])
    if arr.size == 0:
        st.info(f"No data to plot for {title}")
        return
    if MATPLOTLIB_AVAILABLE and plt is not None:
        fig, ax = plt.subplots()
        ax.hist(arr, bins=bins)
        ax.set_title(title)
        st.pyplot(fig)
    else:
        df = pd.DataFrame({"value": arr})
        st.bar_chart(df["value"].value_counts().sort_index())

# ------------------------
# Embedding Manager
# ------------------------
class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    def _load_model(self):
        try:
            st.write(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            st.write(f"Embedding model loaded. Dim: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            st.error(f"Failed to load embedding model: {e}")
            raise
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        if not self.model:
            raise ValueError("Embedding model not loaded")
        emb = self.model.encode(texts, show_progress_bar=False)
        return np.array(emb)

# ------------------------
# VectorStore wrapper (Chroma)
# ------------------------
class VectorStore:
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "./vector_store"):
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()
    def _initialize_store(self):
        try:
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF doc embeddings"}
            )
            st.write(f"Vector store ready. Collection: {self.collection_name} (count={self.collection.count()})")
        except Exception as e:
            st.error(f"Chroma init error: {e}")
            raise
    def add_documents(self, docs: List[Any], embeddings: np.ndarray):
        if len(docs) != len(embeddings):
            raise ValueError("Docs/embeddings length mismatch")
        ids, documents, metadatas, emb_list = [], [], [], []
        for doc, emb in zip(docs, embeddings):
            chunk_hash = sha256_text(doc.page_content)
            file_hash = doc.metadata.get("file_hash")
            doc_id = f"{file_hash}_{chunk_hash}" if file_hash else f"{chunk_hash}"
            ids.append(doc_id)
            documents.append(doc.page_content)
            md = dict(doc.metadata)
            md["chunk_hash"] = chunk_hash
            md["content_length"] = len(doc.page_content)
            metadatas.append(md)
            emb_list.append(emb.tolist())
        try:
            self.collection.delete(ids=ids)
        except Exception:
            pass
        try:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=emb_list
            )
            st.write(f"Indexed {len(ids)} chunks. Collection count now: {self.collection.count()}")
        except Exception as e:
            st.error(f"Error adding to Chroma: {e}")
            raise
    def get_collection(self):
        return self.collection

# ------------------------
# Sparse BM25 index (for sparse-first prefilter)
# ------------------------
class SparseBM25Index:
    def __init__(self):
        self.bm25 = None
        self.docs: List[Dict[str, Any]] = []

    def build(self, docs: List[Any], embeddings: np.ndarray):
        """Build BM25 index over chunk texts + keep ids / metadata / embeddings."""
        if not BM25_AVAILABLE or BM25Okapi is None:
            return
        tokenized_corpus = []
        self.docs = []
        for d, emb in zip(docs, embeddings):
            text = d.page_content
            tokens = tokenize_simple(text)
            if not tokens:
                tokens = [""]
            tokenized_corpus.append(tokens)
            chunk_hash = sha256_text(text)
            file_hash = d.metadata.get("file_hash")
            doc_id = f"{file_hash}_{chunk_hash}" if file_hash else f"{chunk_hash}"
            md = dict(d.metadata)
            md["chunk_hash"] = chunk_hash
            md["content_length"] = len(text)
            self.docs.append({
                "id": doc_id,
                "content": text,
                "metadata": md,
                "emb": emb
            })
        self.bm25 = BM25Okapi(tokenized_corpus)

    def is_ready(self) -> bool:
        return self.bm25 is not None and len(self.docs) > 0

    def get_top_n(self, query: str, n: int) -> List[Dict[str, Any]]:
        """Return top-n docs by BM25 score, with 'sparse_score' included."""
        if not self.is_ready():
            return []
        tokens = tokenize_simple(query)
        if not tokens:
            tokens = [""]
        scores = self.bm25.get_scores(tokens)
        n = min(n, len(self.docs))
        if n <= 0:
            return []
        idx_sorted = np.argsort(scores)[::-1][:n]
        results = []
        for i in idx_sorted:
            doc = dict(self.docs[i])  # shallow copy
            doc["sparse_score"] = float(scores[i])
            results.append(doc)
        return results

# ------------------------
# RAG Retriever (hybrid with sparse-first + dense + cross)
# ------------------------
def ensure_session_cache():
    if "cross_cache" not in st.session_state:
        st.session_state["cross_cache"] = {}
    if "last_index_hashes" not in st.session_state:
        st.session_state["last_index_hashes"] = {}

class RAGRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_manager: EmbeddingManager,
        cross_encoder=None,
        prefilter_multiplier: int = 10,
        prefilter_min: int = 50,
        pref_m_cap: int = 200,
        cross_batch_size: int = 16,
        top_m_for_cross: int = 50,
        sparse_index: Optional[SparseBM25Index] = None,
        use_sparse_prefilter: bool = False,
        sparse_k: int = 200
    ):
        self.vs = vector_store
        self.emb_mgr = embedding_manager
        self.cross = cross_encoder
        self.prefilter_multiplier = prefilter_multiplier
        self.prefilter_min = prefilter_min
        self.pref_m_cap = pref_m_cap
        self.cross_batch_size = cross_batch_size
        self.top_m_for_cross = top_m_for_cross
        self.sparse_index = sparse_index
        self.use_sparse_prefilter = use_sparse_prefilter
        self.sparse_k = sparse_k

    def retrieve(self, query: str, top_k: int = 5):
        ensure_session_cache()
        st.write(f"Query: {query} | top_k={top_k}")
        q_emb = self.emb_mgr.generate_embeddings([query])[0]

        candidates: List[Dict[str, Any]] = []

        # -------- sparse-first prefilter ----------
        if self.use_sparse_prefilter and self.sparse_index is not None and self.sparse_index.is_ready():
            candidates = self.sparse_index.get_top_n(query, self.sparse_k)
            st.write(f"Sparse BM25 prefilter active: {len(candidates)} candidate chunks selected.")
        # ------------------------------------------

        # If sparse candidates unavailable, fallback to dense retrieval from Chroma
        if not candidates:
            candidate_n = int(max(top_k * self.prefilter_multiplier, self.prefilter_min))
            candidate_n = min(candidate_n, self.pref_m_cap)
            try:
                raw = self.vs.collection.query(
                    query_embeddings=[q_emb.tolist()],
                    n_results=candidate_n,
                    include=["documents", "metadatas", "embeddings", "distances"]
                )
            except Exception as e:
                st.error(f"Chroma query error: {e}")
                return []

            docs = raw.get("documents", [[]])[0] if "documents" in raw else raw.get("documents", [[]])[0]
            metas = raw.get("metadatas", [[]])[0] if "metadatas" in raw else []
            emb_arr = np.array(raw.get("embeddings", [[]])[0]) if raw.get("embeddings") else np.array([])
            ids = raw.get("ids", [[]])[0] if raw.get("ids") else []

            if len(docs) == 0:
                st.write("No candidates found.")
                return []

            for i, doc_text in enumerate(docs):
                cid = ids[i] if i < len(ids) else sha256_text(doc_text)
                md = metas[i] if i < len(metas) else {}
                emb_vec = emb_arr[i] if emb_arr.size else None
                candidates.append({
                    "id": cid,
                    "content": doc_text,
                    "metadata": md,
                    "emb": emb_vec,
                    "sparse_score": None
                })

        if not candidates:
            st.write("No candidates available after retrieval.")
            return []

        # ensure we have embeddings for all candidates
        if candidates[0].get("emb") is not None:
            emb_arr = np.vstack([c["emb"] for c in candidates])
        else:
            texts = [c["content"] for c in candidates]
            emb_arr = self.emb_mgr.generate_embeddings(texts)
            for i, e in enumerate(emb_arr):
                candidates[i]["emb"] = e

        # compute cosine similarities
        try:
            qn = q_emb / (np.linalg.norm(q_emb) + 1e-9)
            emb_norms = emb_arr / (np.linalg.norm(emb_arr, axis=1, keepdims=True) + 1e-9)
            sims = np.dot(emb_norms, qn)
        except Exception:
            sims = cosine_similarity([q_emb], emb_arr)[0]

        for i, s in enumerate(sims):
            candidates[i]["cosine_score"] = float(s)

        # prefilter to m (candidates for cross)
        prefilter_m = min(len(candidates), max(top_k * 5, min(self.top_m_for_cross, len(candidates))))
        order = np.argsort(-sims)[:prefilter_m]
        prefiltered: List[Dict[str, Any]] = []
        for idx in order:
            c = candidates[idx]
            prefiltered.append({
                "id": c["id"],
                "content": c["content"],
                "metadata": c.get("metadata", {}),
                "cosine_score": float(c.get("cosine_score", 0.0)),
                "emb": c.get("emb"),
                "sparse_score": float(c.get("sparse_score", 0.0)) if c.get("sparse_score") is not None else None
            })

        # cross-encoder rerank but limit to top_m_for_cross
        if self.cross is not None and len(prefiltered) > 0:
            to_score = prefiltered[: self.top_m_for_cross]
            query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
            scores = [None] * len(prefiltered)
            to_compute = []
            # check cache
            for idx, p in enumerate(to_score):
                cache_key = f"{query_hash}:{p['id']}"
                cached = st.session_state["cross_cache"].get(cache_key)
                if cached is not None:
                    scores[idx] = cached
                else:
                    to_compute.append((idx, cache_key, p["content"]))
            # batch predict
            if to_compute:
                batch_inputs = [(query, doc_text) for (_, _, doc_text) in to_compute]
                for i in range(0, len(batch_inputs), self.cross_batch_size):
                    sub = batch_inputs[i:i + self.cross_batch_size]
                    try:
                        out = self.cross.predict(sub)
                    except Exception:
                        try:
                            combined = [f"{q} [SEP] {d}" for q, d in sub]
                            out = self.cross.predict(combined)
                        except Exception:
                            out = [0.0] * len(sub)
                    for (slot, cache_key, _), val in zip(to_compute[i:i + self.cross_batch_size], out):
                        float_val = float(val)
                        scores[slot] = float_val
                        st.session_state["cross_cache"][cache_key] = float_val
            # merge cross scores back into prefiltered
            for i, p in enumerate(prefiltered):
                if i < len(scores) and scores[i] is not None:
                    p["cross_score"] = float(scores[i])
                else:
                    p["cross_score"] = float(p.get("cosine_score", 0.0))
            ranked = sorted(
                prefiltered,
                key=lambda x: x.get("cross_score", x.get("cosine_score", 0.0)),
                reverse=True
            )[:top_k]
            final = []
            for rank, item in enumerate(ranked, start=1):
                final.append({
                    "id": item["id"],
                    "content": item["content"],
                    "metadata": item.get("metadata", {}),
                    "similarity_score": float(item.get("cross_score", item.get("cosine_score", 0.0))),
                    "rank": rank,
                    "emb": item.get("emb")
                })
            return final
        else:
            # fallback to cosine top_k
            top_idx = np.argsort(-sims)[:top_k]
            final = []
            for rank, i in enumerate(top_idx, start=1):
                c = candidates[i]
                final.append({
                    "id": c["id"],
                    "content": c["content"],
                    "metadata": c.get("metadata", {}),
                    "similarity_score": float(c.get("cosine_score", 0.0)),
                    "rank": rank,
                    "emb": c.get("emb")
                })
            return final

# ------------------------
# Groq LLM wrapper
# ------------------------
class GroqLLM:
    def __init__(self, model_name: str = "llama-3.1-8b-instant", api_key: str = None, max_context_tokens: int = 4096):
        if not LANGCHAIN_AVAILABLE or ChatGroq is None or HumanMessage is None:
            raise RuntimeError(
                "LangChain / ChatGroq not available. "
                "Install `langchain-groq`, `langchain`, and `langchain-community`."
            )
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key required")
        self.llm = ChatGroq(
            groq_api_key=self.api_key,
            model_name=self.model_name,
            temperature=0.1,
            max_tokens=1024
        )
        self.max_context_tokens = max_context_tokens

    def generate_response(self, query: str, context: str, prompt_instructions: Optional[str] = None) -> str:
        extra = (prompt_instructions or "").strip()
        prompt = f"""You are a helpful assistant. Use only the provided context to answer the question.
Context:
{context}

Question: {query}

Answer concisely and cite sources if present. If insufficient information, say you do not know."""
        if extra:
            prompt = extra + "\n\n" + prompt
        try:
            messages = [HumanMessage(content=prompt)]
            resp = self.llm.invoke(messages)
            return resp.content
        except Exception as e:
            return f"LLM call error: {e}"

# ------------------------
# PDF processing & splitting
# ------------------------
def process_pdf(pdf_path: str) -> List[Any]:
    if not LANGCHAIN_AVAILABLE:
        st.error("langchain/pdf loader not available in environment.")
        return []
    st.write(f"Loading PDF: {Path(pdf_path).name}")
    try:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
    except Exception as e:
        st.error(f"PDF loader error: {e}")
        return []
    try:
        file_hash = sha256_file(pdf_path)
    except Exception:
        file_hash = None
    for d in docs:
        d.metadata["source_file"] = Path(pdf_path).name
        d.metadata["file_type"] = "pdf"
        if file_hash:
            d.metadata["file_hash"] = file_hash
    st.write(f"Loaded {len(docs)} pages")
    return docs

def split_documents(documents: List[Any], chunk_size: int = 800, chunk_overlap: int = 128, method: str = "Recursive") -> List[Any]:
    """
    Split documents using different chunking strategies.
    
    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        method: Chunking method - "Recursive", "Hybrid", or "Fixed-size"
    
    Returns:
        List of split documents
    """
    if RecursiveCharacterTextSplitter is None:
        out = []
        for d in documents:
            out.append(d)
        return out
    
    # Define length function (token-aware if tiktoken is available)
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            def length_fn(text: str) -> int:
                return len(enc.encode(text))
        except Exception:
            def length_fn(text: str) -> int:
                return max(1, len(text) // 4)
    else:
        def length_fn(text: str) -> int:
            return max(1, len(text) // 4)
    
    if method == "Recursive":
        # Recursive chunking: splits on separators recursively
        st.write(f"Using Recursive chunking strategy (chunk_size={chunk_size}, overlap={chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = splitter.split_documents(documents)
        st.write(f"Split into {len(split_docs)} chunks using Recursive method")
        return split_docs
    
    elif method == "Hybrid":
        # Hybrid chunking: combine semantic boundaries with fixed-size chunks
        st.write(f"Using Hybrid chunking strategy (chunk_size={chunk_size}, overlap={chunk_overlap})")
        
        # First pass: split on paragraph boundaries (semantic)
        semantic_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size * 2,  # Larger initial chunks
            chunk_overlap=0,
            length_function=length_fn,
            separators=["\n\n", "\n"]  # Only paragraph/line breaks
        )
        semantic_chunks = semantic_splitter.split_documents(documents)
        
        # Second pass: if chunks are still too large, split them further with overlap
        final_chunks = []
        for doc in semantic_chunks:
            doc_len = length_fn(doc.page_content)
            if doc_len > chunk_size:
                # Split large semantic chunks into smaller fixed-size chunks
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    length_function=length_fn,
                    separators=[" ", ""]
                )
                sub_docs = sub_splitter.split_documents([doc])
                final_chunks.extend(sub_docs)
            else:
                final_chunks.append(doc)
        
        st.write(f"Split into {len(final_chunks)} chunks using Hybrid method")
        return final_chunks
    
    elif method == "Fixed-size":
        # Fixed-size chunking: simple character-based splitting
        st.write(f"Using Fixed-size chunking strategy (chunk_size={chunk_size}, overlap={chunk_overlap})")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
            separators=[""]  # No semantic separators, just split at size
        )
        split_docs = splitter.split_documents(documents)
        st.write(f"Split into {len(split_docs)} chunks using Fixed-size method")
        return split_docs
    
    else:
        # Default to recursive if unknown method
        st.warning(f"Unknown chunking method '{method}', defaulting to Recursive")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_fn,
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = splitter.split_documents(documents)
        st.write(f"Split into {len(split_docs)} chunks")
        return split_docs

def assemble_context(
    chunks: List[Dict[str, Any]],
    reserved_answer_tokens: int = 512,
    max_context_tokens: int = 4096
):
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
            def tok_len(text): return len(enc.encode(text))
        except Exception:
            def tok_len(text): return max(1, len(text) // 4)
    else:
        def tok_len(text): return max(1, len(text) // 4)
    selected = []
    token_sum = 0
    budget = max_context_tokens - reserved_answer_tokens
    for c in chunks:
        t = tok_len(c["content"])
        if token_sum + t > budget:
            continue
        selected.append(c)
        token_sum += t
    if len(selected) == 0 and chunks:
        c0 = chunks[0]
        approx_chars = max(100, int((budget // 1) * 4))
        truncated = c0["content"][:approx_chars]
        selected = [{
            "id": c0["id"],
            "content": truncated,
            "metadata": c0["metadata"],
            "similarity_score": c0.get("similarity_score", 0)
        }]
    parts = []
    for idx, c in enumerate(selected, start=1):
        src = c["metadata"].get("source_file", "unknown")
        chunk_hash = c["metadata"].get("chunk_hash", "")
        parts.append(f"[{idx}] Source: {src} (chunk:{chunk_hash})\n{c['content']}\n")
    return "\n\n".join(parts), selected

def map_sentences_to_chunks(
    answer: str,
    used_chunks: List[Dict[str, Any]],
    embedding_manager: EmbeddingManager
):
    sents = simple_sent_tokenize(answer)
    if not sents or not used_chunks:
        return []
    sent_embs = embedding_manager.generate_embeddings(sents)
    if used_chunks and used_chunks[0].get("emb") is not None:
        chunk_embs = np.vstack([c.get("emb") for c in used_chunks])
    else:
        chunk_embs = embedding_manager.generate_embeddings([c["content"] for c in used_chunks])
    sims = cosine_similarity(sent_embs, chunk_embs)
    res = []
    for i, s in enumerate(sents):
        row = sims[i]
        best_idx = int(np.argmax(row))
        res.append({
            "sentence": s,
            "best_chunk_id": used_chunks[best_idx].get("id"),
            "best_source": used_chunks[best_idx]["metadata"].get("source_file", "unknown"),
            "best_chunk_hash": used_chunks[best_idx]["metadata"].get("chunk_hash", ""),
            "best_similarity": float(row[best_idx])
        })
    return res

# Semantic scoring: BERTScore optionally, else embedding cosine fallback
def semantic_scores_for_retrieved(
    gold: str,
    retrieved_texts: List[str],
    embedding_manager: EmbeddingManager,
    use_bertscore: bool
):
    method = "bert_score"
    start = time.time()
    scores = []
    if use_bertscore and BERTSCORE_AVAILABLE and len(retrieved_texts) > 0:
        try:
            refs = [gold] * len(retrieved_texts)
            P, R, F = bert_score_fn(retrieved_texts, refs, lang="en", rescale_with_baseline=True)
            scores = [float(x) for x in F]
            elapsed = time.time() - start
            return scores, method, elapsed
        except Exception:
            method = "bert_score_failed"
    # fallback embedding cosine
    method = "embedding_cosine"
    try:
        texts = [gold] + retrieved_texts
        embs = embedding_manager.generate_embeddings(texts)
        if embs.shape[0] >= 2:
            gold_emb = embs[0]
            ret_embs = embs[1:]
            sims = cosine_similarity([gold_emb], ret_embs)[0]
            scores = [float(s) for s in sims]
        else:
            scores = [0.0] * len(retrieved_texts)
    except Exception:
        scores = [0.0] * len(retrieved_texts)
    elapsed = time.time() - start
    return scores, method, elapsed

# ------------------------
# Streamlit App
# ------------------------
st.set_page_config(page_title="Enhanced RAG (Hybrid Sparse+Dense) + Eval", layout="wide")
st.title("AI ASSISTED TENDER DOCUMENT ANALYSIS USING RAG FRAMEWORK")
st.title("Hybrid Retrieval+Sentence-level Citations + Evaluation")

ensure_session_cache()

# Sidebar controls
groq_api_key = st.sidebar.text_input("Groq API Key (for LLM answers & eval)", type="password")
embed_model_name = st.sidebar.text_input("Embedding model", value="all-MiniLM-L6-v2")
cross_model_name = st.sidebar.text_input("Cross-Encoder model (optional)", value="cross-encoder/ms-marco-MiniLM-L-6-v2")
use_cross = st.sidebar.checkbox("Use Cross-Encoder rerank (if available)", value=True)

# NEW: Chunking strategy controls
st.sidebar.markdown("### Document Chunking Strategy")
chunking_method = st.sidebar.selectbox(
    "Chunking Method",
    ["Recursive", "Hybrid", "Fixed-size"],
    help="Recursive: Split on separators recursively. Hybrid: Combine semantic and fixed-size. Fixed-size: Simple fixed-size chunks."
)
chunk_size = st.sidebar.number_input(
    "Chunk size (tokens/characters)",
    value=800, min_value=100, max_value=2000, step=100
)
chunk_overlap = st.sidebar.number_input(
    "Chunk overlap",
    value=128, min_value=0, max_value=500, step=32
)

# NEW: Sparse prefilter controls
st.sidebar.markdown("### Sparse BM25 Prefilter (Option C)")
use_sparse_prefilter = st.sidebar.checkbox("Use sparse BM25 prefilter (if available)", value=True)
sparse_k = st.sidebar.number_input(
    "BM25 top-N candidates for dense stage",
    value=200, min_value=10, max_value=2000, step=10
)

max_context_tokens = st.sidebar.number_input("LLM context tokens", value=4096, step=512)
reserved_answer_tokens = st.sidebar.number_input("Reserved tokens for answer", value=512, step=64)
top_k_default = st.sidebar.number_input("Top-K results to return (UI default)", value=3, min_value=1, max_value=20)
pref_m_cap = st.sidebar.number_input("Prefilter M cap (Chroma n_results max)", value=200, min_value=10, max_value=1000)
top_m_for_cross = st.sidebar.number_input("Top-M candidates for cross-encoder (M)", value=50, min_value=1, max_value=500)
cross_batch_size = st.sidebar.number_input("Cross-encoder batch size", value=16, min_value=1, max_value=128)

# Answer display mode
st.sidebar.markdown("### Answer display mode")
answer_display_mode = st.sidebar.selectbox(
    "Choose how to show sources",
    ["Answer only", "Answer + inline sources", "Answer + sources in expander"]
)

# Evaluation controls
st.sidebar.markdown("## Evaluation options")
eval_csv = st.sidebar.file_uploader("Upload QA CSV (id,question,answer)", type=["csv"])
eval_rows = st.sidebar.number_input(
    "Number of rows to evaluate (0 = all)",
    min_value=0,
    value=0,
    step=1
)
eval_max_k = st.sidebar.number_input(
    "Evaluate up to K (Recall@K / Precision@K) - use K",
    value=5,
    min_value=1,
    max_value=50
)
eval_use_bertscore = st.sidebar.checkbox(
    "Use BERTScore for semantic matching (fallback to cosine if unavailable)",
    value=True
)
eval_run_llm = st.sidebar.checkbox(
    "Run LLM to compute EM/F1 (expensive, requires API key)",
    value=False
)

# NEW: toggle to compare dense-only vs sparse+dense retrieval
compare_dense_sparse = st.sidebar.checkbox(
    "Compare dense-only vs sparse+dense in evaluation",
    value=False
)

prompt_instructions = st.sidebar.text_area(
    "Optional extra prompt instructions (prepended to LLM prompt)",
    value="",
    help="Small instruction text such as 'Return answer only — do not include sources.'"
)

if eval_run_llm and not groq_api_key:
    st.sidebar.warning(
        "LLM evaluation requested but Groq API key missing — LLM evaluation disabled until key provided."
    )

if use_sparse_prefilter and not BM25_AVAILABLE:
    st.sidebar.warning(
        "Sparse BM25 prefilter requested but `rank_bm25` is not installed. "
        "Run `pip install rank-bm25` to enable."
    )

# File uploader
uploaded_files = st.file_uploader(
    "Upload one or more PDF files",
    type="pdf",
    accept_multiple_files=True
)

# Process uploaded PDFs and index
if uploaded_files and len(uploaded_files) > 0:
    # save temps
    tmp_paths = []
    for f in uploaded_files:
        tmp_name = f"uploaded_{uuid.uuid4().hex[:8]}.pdf"
        with open(tmp_name, "wb") as of:
            of.write(f.getbuffer())
        tmp_paths.append(tmp_name)
    st.info(f"Saved {len(tmp_paths)} PDF(s) for processing.")
    with st.spinner("Processing PDFs..."):
        all_pages = []
        for p in tmp_paths:
            pages = process_pdf(p)
            if pages:
                all_pages.extend(pages)
        if not all_pages:
            st.error("No pages loaded.")
        else:
            chunks = split_documents(all_pages, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), method=chunking_method)

            @st.cache_resource
            def _emb_mgr(name):
                return EmbeddingManager(model_name=name)

            @st.cache_resource
            def _vs(name, path):
                return VectorStore(collection_name=name, persist_directory=path)

            embedding_manager = _emb_mgr(embed_model_name)
            vector_store = _vs("pdf_documents", "./vector_store")
            st.write("Vector store count:", vector_store.collection.count())
            reindex = st.button("Index / Re-index uploaded PDFs")

            bm25_index: Optional[SparseBM25Index] = None

            if vector_store.collection.count() == 0 or reindex:
                st.info("Indexing (may take time)...")
                texts = [d.page_content for d in chunks]
                batch = 64
                emb_batches = []
                for i in range(0, len(texts), batch):
                    emb = embedding_manager.generate_embeddings(texts[i:i+batch])
                    emb_batches.append(emb)
                all_emb = np.vstack(emb_batches) if emb_batches else np.array([])
                if all_emb.size == 0:
                    st.error("Embeddings generation failed.")
                else:
                    try:
                        vector_store.add_documents(chunks, all_emb)
                        st.success("Indexing complete.")
                        for p in tmp_paths:
                            try:
                                st.session_state.setdefault("last_index_hashes", {})[os.path.basename(p)] = sha256_file(p)
                            except Exception:
                                st.session_state["last_index_hashes"][os.path.basename(p)] = None
                        # Build BM25 index if requested and available
                        if use_sparse_prefilter and BM25_AVAILABLE:
                            bm25_index = SparseBM25Index()
                            bm25_index.build(chunks, all_emb)
                            if bm25_index.is_ready():
                                st.success(f"Sparse BM25 index built over {len(chunks)} chunks.")
                            else:
                                st.warning("BM25 index could not be built; falling back to dense-only retrieval.")
                        elif use_sparse_prefilter and not BM25_AVAILABLE:
                            st.warning(
                                "BM25 prefilter requested but `rank_bm25` is not installed. "
                                "Install it to enable sparse prefilter."
                            )
                    except Exception as e:
                        st.error(f"Indexing failed: {e}")
            else:
                st.info("Using existing vector store. Press Index/Re-index to reindex.")
                # no BM25 index in this branch (we don't have chunk objects + embeddings now)
                bm25_index = None

            # load cross encoder optionally
            cross_encoder = None
            if use_cross and CROSS_ENCODER_AVAILABLE:
                @st.cache_resource
                def _load_cross(name):
                    try:
                        return CrossEncoder(name)
                    except Exception:
                        return None
                cross_encoder = _load_cross(cross_model_name)
                if cross_encoder is None:
                    st.warning("Cross-encoder not loaded; falling back to cosine.")
            elif use_cross and not CROSS_ENCODER_AVAILABLE:
                st.warning("Cross-Encoder requested but package not installed; falling back to cosine.")

            retriever = RAGRetriever(
                vector_store,
                embedding_manager,
                cross_encoder,
                prefilter_multiplier=10,
                prefilter_min=50,
                pref_m_cap=int(pref_m_cap),
                cross_batch_size=int(cross_batch_size),
                top_m_for_cross=int(top_m_for_cross),
                sparse_index=bm25_index,
                use_sparse_prefilter=use_sparse_prefilter and (bm25_index is not None and bm25_index.is_ready()),
                sparse_k=int(sparse_k)
            )

            # Initialize LLM if API key present
            groq_llm = None
            if groq_api_key:
                try:
                    groq_llm = GroqLLM(
                        api_key=groq_api_key,
                        max_context_tokens=int(max_context_tokens)
                    )
                except Exception as e:
                    st.error(f"LLM init error: {e}")
                    groq_llm = None

            # -------------------------
            # Interactive Query UI
            # -------------------------
            st.header("Query your Documents")
            q_in = st.text_input("Ask a question (interactive retrieval):")
            top_k = int(top_k_default)
            if q_in:
                with st.spinner("Retrieving and generating answer..."):
                    t0 = time.time()
                    results = retriever.retrieve(q_in, top_k=top_k)
                    t_retr = time.time() - t0
                    ctx_text, used_chunks = assemble_context(
                        results,
                        reserved_answer_tokens=int(reserved_answer_tokens),
                        max_context_tokens=int(max_context_tokens)
                    )
                    if ctx_text and groq_llm is not None:
                        t1 = time.time()
                        raw_ans = groq_llm.generate_response(
                            q_in,
                            ctx_text,
                            prompt_instructions=prompt_instructions
                        )
                        t_gen = time.time() - t1
                    elif ctx_text:
                        raw_ans = "LLM not initialized (no API key or LLM error)."
                        t_gen = 0.0
                    else:
                        raw_ans = "No context available to answer the query."
                        t_gen = 0.0
                    clean_ans = strip_provenance(raw_ans)
                    st.subheader("Answer")
                    if answer_display_mode == "Answer only":
                        st.write(clean_ans)
                    elif answer_display_mode == "Answer + inline sources":
                        st.write(raw_ans)
                    else:
                        st.write(clean_ans)
                        with st.expander("Sources used (click to expand)"):
                            if used_chunks:
                                for r in used_chunks:
                                    src = r.get("metadata", {}).get("source_file", "unknown")
                                    ch = r.get("metadata", {}).get("chunk_hash", "")
                                    st.caption(
                                        f"src: {src} | chunk: {ch} | score: {r.get('similarity_score',0.0):.3f}"
                                    )
                                    st.write(r["content"][:400] + ("..." if len(r["content"]) > 400 else ""))
                            else:
                                st.info("No sources used.")
                    st.info(
                        f"Retrieval time: {t_retr:.3f}s | "
                        f"Generation time: {t_gen:.3f}s | "
                        f"Total: {t_retr + t_gen:.3f}s"
                    )
                    # sentence mapping
                    sent_map = map_sentences_to_chunks(
                        clean_ans if answer_display_mode != "Answer + inline sources" else raw_ans,
                        used_chunks,
                        embedding_manager
                    )
                    if sent_map:
                        st.subheader("Sentence-level citations")
                        for m in sent_map:
                            st.write(f"- {m['sentence']}")
                            st.caption(
                                f"-> {m['best_source']} | chunk:{m['best_chunk_hash']} | "
                                f"sim:{m['best_similarity']:.3f}"
                            )

            # -------------------------
            # Evaluation block
            # -------------------------
            st.header("Evaluation")
            st.write("Upload a QA CSV in the sidebar (columns: id,question,answer). Then run evaluation.")
            eval_df = None
            if eval_csv is not None:
                try:
                    eval_df = pd.read_csv(eval_csv)
                    st.write(f"Loaded {len(eval_df)} rows in evaluation CSV.")
                    if eval_rows and eval_rows > 0:
                        eval_df = eval_df.iloc[:eval_rows]
                        st.write(f"Evaluating first {len(eval_df)} rows.")
                except Exception as e:
                    st.error(f"Could not read eval CSV: {e}")
                    eval_df = None

            if eval_df is not None and st.button("Run Evaluation"):
                max_k = int(eval_max_k)
                use_bertscore = bool(eval_use_bertscore)

                # comparison toggle logic
                do_compare_dense_sparse = bool(compare_dense_sparse)
                if do_compare_dense_sparse and (bm25_index is None or not bm25_index.is_ready()):
                    st.warning(
                        "Dense vs sparse+dense comparison requested, but BM25 index is not available. "
                        "Please re-index with sparse prefilter enabled. Proceeding with single-mode evaluation."
                    )
                    do_compare_dense_sparse = False

                # LLM usage
                run_llm = bool(eval_run_llm) and (groq_llm is not None)
                if eval_run_llm and groq_llm is None:
                    st.warning(
                        "LLM evaluation was requested but GroqLLM is not initialized "
                        "(missing key or error). Proceeding with retrieval-only metrics."
                    )

                # build retrievers for evaluation
                if do_compare_dense_sparse:
                    retriever_dense_eval = RAGRetriever(
                        vector_store,
                        embedding_manager,
                        cross_encoder,
                        prefilter_multiplier=10,
                        prefilter_min=50,
                        pref_m_cap=int(pref_m_cap),
                        cross_batch_size=int(cross_batch_size),
                        top_m_for_cross=int(top_m_for_cross),
                        sparse_index=None,
                        use_sparse_prefilter=False,
                        sparse_k=int(sparse_k)
                    )
                    retriever_sparse_eval = RAGRetriever(
                        vector_store,
                        embedding_manager,
                        cross_encoder,
                        prefilter_multiplier=10,
                        prefilter_min=50,
                        pref_m_cap=int(pref_m_cap),
                        cross_batch_size=int(cross_batch_size),
                        top_m_for_cross=int(top_m_for_cross),
                        sparse_index=bm25_index,
                        use_sparse_prefilter=True,
                        sparse_k=int(sparse_k)
                    )
                    retriever_for_llm = retriever_sparse_eval  # use sparse+dense for LLM
                else:
                    retriever_dense_eval = None
                    retriever_sparse_eval = None
                    retriever_for_llm = retriever  # use current config

                # helper to compute retrieval stats for a candidate list
                def compute_retrieval_stats(cand_list, gold_text: str):
                    retrieved_texts = [c["content"] for c in cand_list]
                    gold_norm = normalize_text(gold_text)

                    # Compute semantic similarity scores for all retrieved texts
                    sem_scores, sem_method, _ = semantic_scores_for_retrieved(
                        gold_text,
                        retrieved_texts,
                        embedding_manager,
                        use_bertscore
                    )
                    
                    # Set relevance threshold: documents with semantic score >= threshold are considered relevant
                    # Using 0.5 as a balanced threshold (can be tuned based on evaluation needs)
                    RELEVANCE_THRESHOLD = 0.5
                    
                    # Ensure we have scores for all retrieved texts (safety check)
                    if len(sem_scores) != len(retrieved_texts):
                        sem_scores = [0.0] * len(retrieved_texts)
                    
                    # Determine relevance for each retrieved document based on semantic similarity
                    relevance = []
                    for i, score in enumerate(sem_scores):
                        # Also check for exact substring match as a strong signal
                        has_exact_match = gold_norm and gold_norm in normalize_text(retrieved_texts[i])
                        # Document is relevant if it has high semantic similarity OR contains exact answer
                        is_relevant = (score >= RELEVANCE_THRESHOLD) or has_exact_match
                        relevance.append(is_relevant)
                    
                    # Find rank of first relevant document
                    found_rank = -1
                    for idx, is_rel in enumerate(relevance):
                        if is_rel:
                            found_rank = idx + 1
                            break

                    def recall_at_k(k):
                        """Recall@k: 1 if at least one relevant doc in top-k, 0 otherwise"""
                        k_eff = min(k, len(relevance))
                        if k_eff <= 0:
                            return 0
                        return 1 if any(relevance[:k_eff]) else 0

                    def prec_at_k(k):
                        """Precision@k: fraction of top-k that are relevant"""
                        k_eff = min(k, len(relevance))
                        if k_eff <= 0:
                            return 0.0
                        topk_relevance = relevance[:k_eff]
                        num_rel = sum(topk_relevance)
                        return num_rel / k_eff

                    r1 = recall_at_k(1)
                    r3 = recall_at_k(3)
                    r5 = recall_at_k(5)

                    p1 = prec_at_k(1)
                    p3 = prec_at_k(3)
                    p5 = prec_at_k(5)

                    if sem_scores:
                        best_sem_idx = int(np.argmax(sem_scores)) + 1
                        best_sem_score = float(np.max(sem_scores))
                    else:
                        best_sem_idx = -1
                        best_sem_score = 0.0

                    return {
                        "found_rank": found_rank,
                        "r1": r1, "r3": r3, "r5": r5,
                        "p1": p1, "p3": p3, "p5": p5,
                        "semantic_method": sem_method,
                        "best_sem_rank": best_sem_idx,
                        "best_sem_score": best_sem_score
                    }

                n = len(eval_df)
                pbar = st.progress(0)
                rows_out = []
                start_eval = time.time()

                for i, r in eval_df.iterrows():
                    qid = str(r.get("id", f"r{i}"))
                    question = str(r.get("question", ""))
                    gold = str(r.get("answer", ""))

                    # retrieval for evaluation
                    rt0 = time.time()
                    if do_compare_dense_sparse:
                        cand_dense = retriever_dense_eval.retrieve(question, top_k=max_k)
                        cand_sparse = retriever_sparse_eval.retrieve(question, top_k=max_k)
                        rt_elapsed = time.time() - rt0

                        stats_dense = compute_retrieval_stats(cand_dense, gold)
                        stats_sparse = compute_retrieval_stats(cand_sparse, gold)

                        cand_for_llm = cand_sparse  # use sparse+dense for LLM metrics
                    else:
                        cand_primary = retriever_for_llm.retrieve(question, top_k=max_k)
                        rt_elapsed = time.time() - rt0

                        stats_primary = compute_retrieval_stats(cand_primary, gold)

                        cand_dense = None
                        cand_sparse = None
                        cand_for_llm = cand_primary

                    # LLM generation metrics (if requested)
                    generated_raw = ""
                    generated_clean = ""
                    em = None; f1 = None; token_prec = None; rouge1_score = None; rougeL_score = None
                    gen_time = 0.0
                    if run_llm:
                        ctx, used = assemble_context(
                            cand_for_llm,
                            reserved_answer_tokens=int(reserved_answer_tokens),
                            max_context_tokens=int(max_context_tokens)
                        )
                        if ctx:
                            gen_start = time.time()
                            generated_raw = groq_llm.generate_response(
                                question,
                                ctx,
                                prompt_instructions=prompt_instructions
                            )
                            gen_time = time.time() - gen_start
                            generated_clean = strip_provenance(generated_raw)
                            emv, f1v = compute_exact_and_f1(generated_clean, gold)
                            em = int(emv)
                            f1 = float(f1v)
                            token_prec = token_precision(generated_clean, gold)
                            rouge1_score = rouge_1(generated_clean, gold)
                            rougeL_score = rouge_l(generated_clean, gold)
                        else:
                            generated_raw = ""
                            generated_clean = ""
                            em = 0; f1 = 0.0; token_prec = 0.0; rouge1_score = 0.0; rougeL_score = 0.0

                    if rouge1_score is None:
                        rouge1_score = 0.0
                    if rougeL_score is None:
                        rougeL_score = 0.0
                    if token_prec is None:
                        token_prec = 0.0

                    row_dict = {
                        "id": qid,
                        "question": question,
                        "gold": gold,
                        "retrieval_time_sec": float(rt_elapsed),
                        "generated_answer_clean": generated_clean,
                        "generated_answer_raw": generated_raw,
                        "generation_time_sec": float(gen_time),
                        "em": em,
                        "f1": f1,
                        "token_precision": token_prec,
                        "rouge_1": rouge1_score,
                        "rouge_l": rougeL_score
                    }

                    if do_compare_dense_sparse:
                        # dense-only metrics
                        row_dict.update({
                            "found_in_retrieval_rank_dense": stats_dense["found_rank"],
                            "recall_at_1_dense": stats_dense["r1"],
                            "recall_at_3_dense": stats_dense["r3"],
                            "recall_at_5_dense": stats_dense["r5"],
                            "precision_at_1_dense": stats_dense["p1"],
                            "precision_at_3_dense": stats_dense["p3"],
                            "precision_at_5_dense": stats_dense["p5"],
                            "semantic_method_dense": stats_dense["semantic_method"],
                            "best_semantic_score_dense": stats_dense["best_sem_score"],
                            "best_semantic_rank_dense": stats_dense["best_sem_rank"],
                        })
                        # sparse+dense metrics
                        row_dict.update({
                            "found_in_retrieval_rank_sparse": stats_sparse["found_rank"],
                            "recall_at_1_sparse": stats_sparse["r1"],
                            "recall_at_3_sparse": stats_sparse["r3"],
                            "recall_at_5_sparse": stats_sparse["r5"],
                            "precision_at_1_sparse": stats_sparse["p1"],
                            "precision_at_3_sparse": stats_sparse["p3"],
                            "precision_at_5_sparse": stats_sparse["p5"],
                            "semantic_method_sparse": stats_sparse["semantic_method"],
                            "best_semantic_score_sparse": stats_sparse["best_sem_score"],
                            "best_semantic_rank_sparse": stats_sparse["best_sem_rank"],
                        })
                    else:
                        row_dict.update({
                            "found_in_retrieval_rank": stats_primary["found_rank"],
                            "recall_at_1": stats_primary["r1"],
                            "recall_at_3": stats_primary["r3"],
                            "recall_at_5": stats_primary["r5"],
                            "precision_at_1": stats_primary["p1"],
                            "precision_at_3": stats_primary["p3"],
                            "precision_at_5": stats_primary["p5"],
                            "semantic_method": stats_primary["semantic_method"],
                            "best_semantic_score": stats_primary["best_sem_score"],
                            "best_semantic_rank": stats_primary["best_sem_rank"],
                        })

                    rows_out.append(row_dict)
                    pbar.progress((i + 1) / n)

                elapsed_eval = time.time() - start_eval
                res_df = pd.DataFrame(rows_out)
                st.success(f"Evaluation finished in {elapsed_eval:.1f}s on {len(res_df)} rows.")

                st.subheader("Aggregated Retrieval Metrics")
                if do_compare_dense_sparse:
                    st.write("**Dense-only retrieval**")
                    st.write(f"- Recall@1: {res_df['recall_at_1_dense'].mean():.3f}")
                    st.write(f"- Recall@3: {res_df['recall_at_3_dense'].mean():.3f}")
                    st.write(f"- Recall@5: {res_df['recall_at_5_dense'].mean():.3f}")
                    st.write(f"- Precision@1: {res_df['precision_at_1_dense'].mean():.3f}")
                    st.write(f"- Precision@3: {res_df['precision_at_3_dense'].mean():.3f}")
                    st.write(f"- Precision@5: {res_df['precision_at_5_dense'].mean():.3f}")
                    st.write("")
                    st.write("**Sparse+dense (BM25 prefilter) retrieval**")
                    st.write(f"- Recall@1: {res_df['recall_at_1_sparse'].mean():.3f}")
                    st.write(f"- Recall@3: {res_df['recall_at_3_sparse'].mean():.3f}")
                    st.write(f"- Recall@5: {res_df['recall_at_5_sparse'].mean():.3f}")
                    st.write(f"- Precision@1: {res_df['precision_at_1_sparse'].mean():.3f}")
                    st.write(f"- Precision@3: {res_df['precision_at_3_sparse'].mean():.3f}")
                    st.write(f"- Precision@5: {res_df['precision_at_5_sparse'].mean():.3f}")
                else:
                    st.write("Current retrieval configuration")
                    st.write(f"- Recall@1: {res_df['recall_at_1'].mean():.3f}")
                    st.write(f"- Recall@3: {res_df['recall_at_3'].mean():.3f}")
                    st.write(f"- Recall@5: {res_df['recall_at_5'].mean():.3f}")
                    st.write(f"- Precision@1: {res_df['precision_at_1'].mean():.3f}")
                    st.write(f"- Precision@3: {res_df['precision_at_3'].mean():.3f}")
                    st.write(f"- Precision@5: {res_df['precision_at_5'].mean():.3f}")

                if run_llm:
                    st.subheader("Aggregated LLM Metrics (using primary pipeline)")
                    st.write(f"- EM: {res_df['em'].dropna().mean():.3f}")
                    st.write(f"- F1: {res_df['f1'].dropna().mean():.3f}")
                    st.write(f"- Token Precision: {res_df['token_precision'].dropna().mean():.3f}")
                    st.write(f"- ROUGE-1: {res_df['rouge_1'].dropna().mean():.3f}")
                    st.write(f"- ROUGE-L: {res_df['rouge_l'].dropna().mean():.3f}")

                st.subheader("Aggregated Semantic (BERTScore/Embedding) stats")
                if do_compare_dense_sparse:
                    st.write(
                        f"- Dense-only semantic methods: "
                        f"{res_df['semantic_method_dense'].value_counts().to_dict()}"
                    )
                    st.write(
                        f"- Dense-only mean best semantic score: "
                        f"{res_df['best_semantic_score_dense'].mean():.3f}"
                    )
                    st.write(
                        f"- Sparse+dense semantic methods: "
                        f"{res_df['semantic_method_sparse'].value_counts().to_dict()}"
                    )
                    st.write(
                        f"- Sparse+dense mean best semantic score: "
                        f"{res_df['best_semantic_score_sparse'].mean():.3f}"
                    )
                else:
                    st.write(
                        f"- semantic method counts: "
                        f"{res_df['semantic_method'].value_counts().to_dict()}"
                    )
                    st.write(
                        f"- mean best semantic score: "
                        f"{res_df['best_semantic_score'].mean():.3f}"
                    )

                st.subheader("Sample per-question results (first 200 rows)")
                st.dataframe(res_df.head(200))
                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download evaluation CSV",
                    data=csv_bytes,
                    file_name="evaluation_results.csv",
                    mime="text/csv"
                )

                st.subheader("Metric Distributions")
                if do_compare_dense_sparse:
                    plot_hist(res_df["best_semantic_score_dense"], "Dense-only best semantic score distribution")
                    plot_hist(res_df["best_semantic_score_sparse"], "Sparse+dense best semantic score distribution")
                    plot_hist(res_df["precision_at_1_dense"], "Dense-only Precision@1 distribution")
                    plot_hist(res_df["precision_at_1_sparse"], "Sparse+dense Precision@1 distribution")
                else:
                    plot_hist(res_df["best_semantic_score"], "Best semantic score distribution")
                    plot_hist(res_df["precision_at_1"], "Precision@1 distribution")
                if run_llm:
                    plot_hist(res_df["f1"].dropna(), "LLM F1 distribution")
                    plot_hist(res_df["rouge_1"].dropna(), "ROUGE-1 distribution")
                    plot_hist(res_df["rouge_l"].dropna(), "ROUGE-L distribution")

            # cleanup temp files
            for p in tmp_paths:
                try:
                    os.remove(p)
                except Exception:
                    pass
else:
    st.info("Upload one or more PDF files to begin (and then index).")

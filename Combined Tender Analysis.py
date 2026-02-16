# MergedRAGSummarizer.py
# 1:1 functional merge of RAG and Summarization logic
# Restructured as top-down for maximum Streamlit stability.

import os
import re
import time
import json
import uuid
import hashlib
import importlib
import pickle
import tempfile
import asyncio
import random
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import streamlit as st
import numpy as np
import pandas as pd
import chromadb
import fitz  # PyMuPDF
import nltk
from google import genai
from docx import Document
import pdfplumber
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------
# Page Configuration (Must be first)
# --------------------------
st.set_page_config(page_title="RAG & Summarizer", layout="wide")

# --------------------------
# Optional & Heavy Imports
# --------------------------
try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:
    CrossEncoder = None
    CROSS_ENCODER_AVAILABLE = False

try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_groq import ChatGroq
    from langchain.schema import HumanMessage
    LANGCHAIN_AVAILABLE = True
except Exception:
    PyPDFLoader = RecursiveCharacterTextSplitter = ChatGroq = HumanMessage = None
    LANGCHAIN_AVAILABLE = False

try:
    from bert_score import score as bert_score_fn
    BERTSCORE_AVAILABLE = True
except Exception:
    bert_score_fn = None
    BERTSCORE_AVAILABLE = False

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except Exception:
    BM25Okapi = None
    BM25_AVAILABLE = False

# --------------------------
# Constants
# --------------------------
MAX_FILE_SIZE_MB = 50
MAX_QUERY_LENGTH = 5000
DEFAULT_RATE_LIMIT_SLEEP = 15.0  # seconds
ZOOM_LEVEL = 1.5
MAX_CHUNK_TOKEN = 4096
DEFAULT_EMBEDDING_DIM = 384

# --------------------------
# Initialization
# --------------------------
@st.cache_resource
def setup_nltk_shared():
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
        return True
    except Exception as e:
        st.warning(f"Failed to download NLTK data: {e}")
        return False

_NLTK_READY = setup_nltk_shared()

# Shared Utils
def sha256_text_shared(text: str) -> str:
    return hashlib.sha256((text or "").strip().encode("utf-8")).hexdigest()

def sha256_file_shared(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def normalize_text_shared(s: str) -> str:
    if s is None: return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize_simple_shared(s: str) -> List[str]:
    return re.findall(r"\w+", (s or "").lower())

_SENTENCE_RE = re.compile(r'(?<=[.!?])\s+')
def simple_sent_tokenize_shared(text: str) -> List[str]:
    sents = [s.strip() for s in _SENTENCE_RE.split(text or "") if s.strip()]
    return sents if sents else ([text.strip()] if (text or "").strip() else [])

_PROV_SPLIT_RE = re.compile(r"\n\s*(\[\d+\]\s*Source:|→\s*Source:|Source:|Source\s*[:\-])", flags=re.IGNORECASE)
def strip_provenance(text: str) -> str:
    if not text: return ""
    parts = _PROV_SPLIT_RE.split(text)
    cleaned = parts[0] if parts else text
    cleaned = re.sub(r"\(chunk:?[^\)]*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\[\(]\s*\d+\s*[\]\)]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def compute_mean_support_score_shared(results: List[Dict[str, Any]]) -> float:
    if not results: return 0.0
    scores = [float(r.get("similarity_score", 0.0)) for r in results]
    return float(np.max(scores)) if scores else 0.0

# --------------------------
# RAG Components
# --------------------------
class EmbeddingManagerRAG:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False)

class VectorStoreRAG:
    def __init__(self, collection_name: str = "pdf_documents", path: str = "./vector_store"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
    def add_documents(self, docs, embs):
        ids = [f"{d.metadata.get('file_hash','na')}_{sha256_text_shared(d.page_content)}" for d in docs]
        self.collection.add(ids=ids, documents=[d.page_content for d in docs], metadatas=[d.metadata for d in docs], embeddings=embs.tolist())

class SparseBM25IndexRAG:
    def __init__(self): self.bm25, self.docs = None, []
    def build(self, docs, embs):
        self.docs = [{"id": sha256_text_shared(d.page_content), "content": d.page_content, "metadata": d.metadata, "emb": e} for d, e in zip(docs, embs)]
        self.bm25 = BM25Okapi([tokenize_simple_shared(d.page_content) for d in docs])
    def is_ready(self): return self.bm25 is not None
    def get_top_n(self, q, n):
        if not self.bm25: return []
        scores = self.bm25.get_scores(tokenize_simple_shared(q))
        idx = np.argsort(scores)[::-1][:n]
        return [dict(self.docs[i], sparse_score=float(scores[i])) for i in idx]

class RAGRetrieverRAG:
    def __init__(self, vs, em, cross=None, sparse=None, use_s=False, s_k=200):
        self.vs, self.em, self.cross, self.sparse, self.use_s, self.s_k = vs, em, cross, sparse, use_s, s_k
    def retrieve(self, q, top_k=5):
        if not q or not q.strip():
            st.warning("Empty query provided. Please enter a valid question.")
            return []
        
        q_emb = self.em.generate_embeddings([q])[0]
        if self.use_s and self.sparse and self.sparse.is_ready():
            cands = self.sparse.get_top_n(q, self.s_k)
        else:
            raw = self.vs.collection.query(query_embeddings=[q_emb.tolist()], n_results=50, include=["documents", "metadatas", "embeddings"])
            if not raw["documents"][0]:
                st.warning("No documents found in the vector store.")
                return []
            cands = [{"id": raw["ids"][0][i], "content": raw["documents"][0][i], "metadata": raw["metadatas"][0][i], "emb": raw["embeddings"][0][i]} for i in range(len(raw["documents"][0]))]
        
        if not cands:
            return []
        
        sims = cosine_similarity([q_emb], [c["emb"] for c in cands])[0]
        for i, s in enumerate(sims): cands[i]["similarity_score"] = float(s)
        res = sorted(cands, key=lambda x: x["similarity_score"], reverse=True)[:top_k]
        return res

class GroqLLMRAG:
    def __init__(self, model="llama-3.1-8b-instant", key=None):
        self.llm = ChatGroq(groq_api_key=key, model_name=model, temperature=0.1)
    def generate_response(self, q, ctx, prompt_instr=""):
        p = f"{prompt_instr}\n\nContext:\n{ctx}\n\nQuestion: {q}"
        return self.llm.invoke([HumanMessage(content=p)]).content

def assemble_context_rag(chunks, max_ctx=4096):
    parts = [f"Source: {c['metadata'].get('source_file')} [{c['id']}]\n{c['content']}" for c in chunks]
    full_text = "\n\n".join(parts)
    # Enforce max_ctx by truncating if necessary
    if len(full_text) > max_ctx * 4:  # Rough approximation: 1 token ≈ 4 chars
        full_text = full_text[:max_ctx * 4]
    return full_text, chunks

def map_sentences_to_chunks_rag(answer, used, em):
    sents = simple_sent_tokenize_shared(answer)
    if not sents or not used: 
        return []
    try:
        s_embs = em.generate_embeddings(sents)
        c_embs = np.vstack([c["emb"] for c in used])
        sims = cosine_similarity(s_embs, c_embs)
        return [{"sentence": s, "source": used[int(np.argmax(sims[i]))]["metadata"].get("source_file"), "score": float(np.max(sims[i]))} for i, s in enumerate(sents)]
    except Exception as e:
        st.warning(f"Failed to map sentences to chunks: {e}")
        return []

# --------------------------
# Summarization Components
# --------------------------
GLOBAL_SUM_CONCURRENCY = asyncio.Semaphore(2)

async def call_gemini_json_sum_async(client, sys, user, model, rpm):
    await asyncio.sleep(DEFAULT_RATE_LIMIT_SLEEP / max(1, rpm))  # Simplistic rate control
    async with GLOBAL_SUM_CONCURRENCY:
        try:
            from google.genai import types
            resp = await client.aio.models.generate_content(model=model, contents=user, config=types.GenerateContentConfig(system_instruction=sys, response_mime_type="application/json"))
            txt = resp.text or "{}"
            if "```json" in txt: txt = txt.split("```json")[1].split("```")[0].strip()
            parsed = json.loads(txt)
            return parsed
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON response: {str(e)}"}
        except Exception as e:
            return {"error": f"API call failed: {str(e)}"}

def extract_pages_sum(path):
    doc = None
    pl_doc = None
    try:
        doc = fitz.open(path)
        p, f = [], []
        with pdfplumber.open(path) as pl_doc:
            for i, page in enumerate(doc):
                t = page.get_text()
                if i < len(pl_doc.pages):
                    tabs = pl_doc.pages[i].extract_tables()
                    for tab in tabs:
                        if tab: t += f"\n\n[TABLE]\n{pd.DataFrame(tab).to_markdown()}\n"
                p.append(t); f.append(len(page.get_images()) > 0)
        return p, f
    finally:
        if doc:
            doc.close()

def semantic_chunk_pages_sum(pages, flags, max_tok=8000, overlap=5):
    flat = []
    for i, p in enumerate(pages):
        if _NLTK_READY:
            try:
                sents = nltk.sent_tokenize(p)
            except Exception:
                # Fallback to simple sentence tokenization if NLTK fails
                sents = simple_sent_tokenize_shared(p)
        else:
            sents = simple_sent_tokenize_shared(p)
        for s in sents:
            flat.append((i+1, s, flags[i]))
    chunks, bs, bp, bf = [], [], [], []
    for p, s, f in flat:
        if sum(len(x)//4 for x in bs + [s]) > max_tok and bs:
            chunks.append({"text": " ".join(bs).strip(), "start_page": min(bp), "has_visual": any(bf), "id": str(uuid.uuid4())[:8]})
            bs, bp, bf = bs[-overlap:] if overlap > 0 else [], bp[-overlap:] if overlap > 0 else [], bf[-overlap:] if overlap > 0 else []
        bs.append(s); bp.append(p); bf.append(f)
    if bs: chunks.append({"text": " ".join(bs).strip(), "start_page": min(bp), "has_visual": any(bf), "id": str(uuid.uuid4())[:8]})
    return [c for c in chunks if c["text"].strip()]

def get_sum_prompts(mode: str):
    if mode == "Compliance Matrix":
        return ("Extract requirements.", "JSON: [{\"item\": \"...\", \"detail\": \"...\", \"page\": 1}]", "Merge matrix.", "JSON: {\"matrix\": [...]}")
    elif mode == "Risk Assessment":
        return ("Flag risks.", "JSON: [{\"clause\": \"...\", \"risk_level\": \"...\", \"page\": 1}]", "Merge risks.", "JSON: {\"risks\": [...]}")
    elif mode == "Entity Dashboard":
        return ("Extract metadata.", "JSON: [{\"category\": \"...\", \"entity\": \"...\", \"page\": 1}]", "Compile Dashboard.", "JSON: {\"dashboard\": {}}")
    else:
        return ("Extract details.", "JSON: [{\"finding\": \"...\", \"page\": 1}]", "Synthesize summary.", "JSON: {\"summary\": \"...\", \"citations\": [pages]}")

def compute_confidence_score_sum(mapped, reduced, q):
    c = {"snippet_coverage": min(1.0, len(mapped)/5.0), "result_coherence": 0.8 if isinstance(reduced, dict) and "error" not in reduced else 0.1, "information_density": 0.5, "citation_confidence": 0.7}
    c["overall_confidence"] = 0.3*c["snippet_coverage"] + 0.3*c["result_coherence"] + 0.4*c["citation_confidence"]
    return c

def render_citation_preview_sum(doc, cites):
    if not cites or not doc: 
        return
    st.markdown("### 📄 Context Preview")
    tabs = st.tabs([f"Page {c['page']}" for c in cites[:5]])
    for i, c in enumerate(cites[:5]):
        with tabs[i]:
            try:
                page = doc.load_page(c['page'] - 1)
                pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM_LEVEL, ZOOM_LEVEL))
                st.image(pix.tobytes("png"))
            except Exception as e:
                st.error(f"Failed to load page {c['page']}: {e}")

# --------------------------
# Main Program logic
# --------------------------
st.sidebar.title("App Navigation")
choice = st.sidebar.radio("Choose Program", ["Simple QA (RAG)", "Context Understanding (Summarization)"])

if choice == "Simple QA (RAG)":
    st.title("AI ASSISTED TENDER DOCUMENT ANALYSIS USING RAG FRAMEWORK")
    with st.sidebar:
        st.header("RAG Settings")
        ga_key = st.text_input("Groq API Key", type="password", key="rgk")
        emb_m = st.text_input("Embedding model", "all-MiniLM-L6-v2", key="rem")
        crs_m = st.text_input("Cross-Encoder", "cross-encoder/ms-marco-MiniLM-L-6-v2", key="rcm")
        u_crs = st.checkbox("Use Cross-Encoder", True, key="rucx")
        u_spa = st.checkbox("Use Sparse BM25", True, key="rusp")
        c_size = st.number_input("Chunk size", 800, key="rcsi")
        c_over = st.number_input("Overlap", 128, key="rcov")
        max_tk = st.number_input("Context Tokens", 4096, key="rctx")
        top_k = st.number_input("Top K UI", 3, key="rtk")
        supp_th = st.slider("Support threshold", 0.1, 1.0, 0.45, key="rsth")

    files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="rfl")
    if files:
        # Validate file sizes
        for f in files:
            file_size_mb = len(f.getvalue()) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File {f.name} exceeds maximum size of {MAX_FILE_SIZE_MB}MB (size: {file_size_mb:.1f}MB)")
                st.stop()
        
        if st.button("Index / Re-index", key="rib"):
            if not LANGCHAIN_AVAILABLE:
                st.error("LangChain libraries are not available. Please install them: pip install langchain langchain-community pypdf")
                st.stop()
            
            with st.spinner("Indexing..."):
                em = EmbeddingManagerRAG(emb_m); vs = VectorStoreRAG()
                all_chunks = []
                for f in files:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                        try:
                            t.write(f.getbuffer())
                            t.flush()
                            h = sha256_file_shared(t.name)
                            ldr = PyPDFLoader(t.name)
                            docs = ldr.load()
                            for d in docs:
                                d.metadata.update({"file_hash": h, "source_file": f.name})
                            all_chunks.extend(RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_over).split_documents(docs))
                        finally:
                            # Ensure temporary file is deleted
                            try:
                                os.unlink(t.name)
                            except Exception:
                                pass
                
                if not all_chunks:
                    st.error("No content could be extracted from the uploaded files.")
                    st.stop()
                
                texts = [c.page_content for c in all_chunks]
                embs = em.generate_embeddings(texts)
                vs.add_documents(all_chunks, embs)
                st.session_state["rvs"], st.session_state["sem"] = vs, em
                
                if u_spa and BM25_AVAILABLE:
                    bm = SparseBM25IndexRAG()
                    bm.build(all_chunks, embs)
                    st.session_state["rbm"] = bm
                elif u_spa and not BM25_AVAILABLE:
                    st.warning("BM25 is not available. Sparse retrieval disabled. Install: pip install rank-bm25")
                
                st.success("Indexing Success!")

    if "rvs" in st.session_state:
        q = st.text_input("Ask a question:", key="rqi")
        if q:
            # Validate query length
            if len(q) > MAX_QUERY_LENGTH:
                st.error(f"Query exceeds maximum length of {MAX_QUERY_LENGTH} characters")
                st.stop()
            
            if not ga_key:
                st.error("Please provide a Groq API Key in the sidebar")
                st.stop()
            
            with st.spinner("Thinking..."):
                # Cache cross-encoder model in session state
                if u_crs and CROSS_ENCODER_AVAILABLE:
                    if "cross_encoder" not in st.session_state:
                        st.session_state["cross_encoder"] = CrossEncoder(crs_m)
                    cr = st.session_state["cross_encoder"]
                else:
                    if u_crs and not CROSS_ENCODER_AVAILABLE:
                        st.warning("Cross-encoder not available. Install: pip install sentence-transformers")
                    cr = None
                
                ret = RAGRetrieverRAG(st.session_state["rvs"], st.session_state["sem"], cr, sparse=st.session_state.get("rbm"), use_s=u_spa)
                res = ret.retrieve(q, top_k)
                
                if not res:
                    st.warning("No relevant documents found for your query. Try rephrasing or check if documents are properly indexed.")
                    st.stop()
                
                ctx, used = assemble_context_rag(res, max_ctx=max_tk)
                llm = GroqLLMRAG(key=ga_key)
                ans = llm.generate_response(q, ctx)
                st.subheader("Answer")
                st.write(strip_provenance(ans))
                supp = compute_mean_support_score_shared(res)
                if supp < supp_th:
                    st.warning(f"Low confidence support: {supp:.2f}")
                else:
                    st.success(f"Strong support: {supp:.2f}")
                s_map = map_sentences_to_chunks_rag(strip_provenance(ans), used, st.session_state["sem"])
                if s_map:
                    with st.expander("Sentence Citations"):
                        for m in s_map:
                            st.write(f"- {m['sentence']} ({m['source']}, {m['score']:.2f})")

else:
    st.title("Tender Analyzer — Gemini Map/Reduce")
    with st.sidebar:
        st.header("SUM Settings")
        gem_key = st.text_input("Google API Key", type="password", key="gka")
        s_obj = st.radio("Objective", ["General Summary", "Compliance Matrix", "Risk Assessment", "Entity Dashboard"], key="sob")
        s_model = st.text_input("Model", value="gemini-2.5-flash", key="smk")
        s_rpm = st.number_input("Target RPM", min_value=1, max_value=4, value=4, key="srp")
        s_batch = st.number_input("Batch size", min_value=10, max_value=50, value=10, key="sbt")
        s_max_tk = st.number_input("Max tokens/chunk", min_value=500, max_value=8000, value=8000, key="smt")
        s_over = st.number_input("Overlap sents", value=5, key="sov")
        s_prompt = st.text_area("Extra prompts", key="sep")

    f_sum = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True, key="sf")
    q_sum = st.text_area("Queries (one per line):", key="sq")
    
    if st.button("Analyze", key="sab"):
        if not f_sum:
            st.error("Please upload at least one PDF file")
            st.stop()
        if not gem_key:
            st.error("Please provide a Google API Key in the sidebar")
            st.stop()
        if not q_sum or not q_sum.strip():
            st.error("Please provide at least one query")
            st.stop()
        
        # Validate file sizes
        for f in f_sum:
            file_size_mb = len(f.getvalue()) / (1024 * 1024)
            if file_size_mb > MAX_FILE_SIZE_MB:
                st.error(f"File {f.name} exceeds maximum size of {MAX_FILE_SIZE_MB}MB (size: {file_size_mb:.1f}MB)")
                st.stop()
        
        client = genai.Client(api_key=gem_key)
        all_p, all_f = [], []
        temp_files = []
        
        try:
            for f in f_sum:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as t:
                    temp_files.append(t.name)
                    t.write(f.getbuffer())
                    t.flush()
                    p, fl = extract_pages_sum(t.name)
                    all_p.extend(p)
                    all_f.extend(fl)
            
            if not all_p:
                st.error("No content could be extracted from the uploaded files")
                st.stop()
            
            chunks = semantic_chunk_pages_sum(all_p, all_f, max_tok=s_max_tk, overlap=s_over)
            
            if not chunks:
                st.error("No chunks could be created from the extracted content")
                st.stop()
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass
        
        async def run_sum():
            ms, mi, rs, ri = get_sum_prompts(s_obj)
            results = []
            instr = s_prompt.strip() + "\n\n" if s_prompt.strip() else ""
            batches = [chunks[i:i+s_batch] for i in range(0, len(chunks), s_batch)]
            for q in q_sum.splitlines():
                if not q.strip(): 
                    continue
                if len(q) > MAX_QUERY_LENGTH:
                    st.warning(f"Skipping query (too long): {q[:100]}...")
                    continue
                t0 = time.time()
                m_tasks = []
                for b in batches:
                    b_txt = '\n'.join([c['text'] for c in b])
                    m_tasks.append(call_gemini_json_sum_async(client, ms, f"{instr}{mi}\nQ:{q}\nC:{b_txt}", s_model, s_rpm))
                mapped = await asyncio.gather(*m_tasks)
                t_m = time.time()
                red = await call_gemini_json_sum_async(client, rs, f"{instr}{ri}\nQ:{q}\nD:{json.dumps(mapped)}", s_model, s_rpm)
                results.append({"query":q, "result":red, "map_t":t_m-t0, "red_t":time.time()-t_m, "mapped":mapped})
            return results

        final_res = asyncio.run(run_sum())
        for r in final_res:
            st.subheader(f"Query: {r['query']}")
            st.caption(f"⏱️ Map: {r['map_t']:.2f}s | Reduce: {r['red_t']:.2f}s")
            
            # Check for errors in result
            if isinstance(r["result"], dict) and "error" in r["result"]:
                st.error(f"Error processing query: {r['result']['error']}")
                continue
            
            st.json(r["result"])
            conf = compute_confidence_score_sum(r['mapped'], r['result'], r['query'])
            st.info(f"Confidence: {conf['overall_confidence']:.2%}")
            
            # Find pages for preview
            pgs = []
            def _f(o):
                if isinstance(o, dict):
                    for k, v in o.items():
                        if k in ["page", "citations"]:
                            if isinstance(v, list): 
                                [pgs.append(x) for x in v if isinstance(x, int)]
                            elif isinstance(v, int): 
                                pgs.append(v)
                        else: 
                            _f(v)
                elif isinstance(o, list): 
                    [_f(x) for x in o]
            _f(r['result'])
            
            if pgs and f_sum:
                try:
                    pdf_doc = fitz.open(stream=f_sum[0].getvalue(), filetype="pdf")
                    render_citation_preview_sum(pdf_doc, [{"page": p} for p in set(pgs)])
                    pdf_doc.close()
                except Exception as e:
                    st.error(f"Failed to load PDF for preview: {e}")


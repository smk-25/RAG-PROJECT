import os
import re
import hashlib
import time
import uuid
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# Hybrid Chunker Class
class HybridChunker:
    def __init__(self, chunk_size=800, chunk_overlap=128, model_name="all-MiniLM-L6-v2"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def tokenize(self, text):
        return len(self.tokenizer.encode(text))

    def create_chunks(self, text):
        words = text.split()
        chunks = []
        current_chunk = []
        tokens_in_chunk = 0
        for word in words:
            word_tokens = self.tokenize(word)
            if tokens_in_chunk + word_tokens <= self.chunk_size:
                current_chunk.append(word)
                tokens_in_chunk += word_tokens
            else:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-self.chunk_overlap:]  # Retain overlap
                tokens_in_chunk = sum(self.tokenize(w) for w in current_chunk)
                current_chunk.append(word)
                tokens_in_chunk += word_tokens
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def refine_chunks(self, chunks, num_clusters=5):
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        if len(chunks) <= num_clusters:
            return chunks
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        chunk_groups = {i: [] for i in range(num_clusters)}
        for i, chunk in enumerate(chunks):
            chunk_groups[labels[i]].append(chunk)
        refined = [" ".join(group) for group in chunk_groups.values()]
        return refined

    def process(self, text):
        chunks = self.create_chunks(text)
        refined = self.refine_chunks(chunks)
        return refined


# Updated App Code
import streamlit as st

st.set_page_config(page_title="Enhanced RAG (Hybrid + Recursive) + Eval", layout="wide")
st.title("AI ASSISTED TENDER DOCUMENT ANALYSIS USING RAG FRAMEWORK")
st.title("Hybrid vs Recursive Chunking")

# Sidebar: Chunking Method Selector
chunking_method = st.sidebar.selectbox("Chunking Method", ["Recursive", "Hybrid"])

# Sidebar settings for chunk size and overlap
st.sidebar.markdown("### Chunking Configuration")
chunk_size = st.sidebar.number_input("Chunk Size", value=800, step=100)
chunk_overlap = st.sidebar.number_input("Chunk Overlap", value=128, step=32)

# Process uploaded PDFs
uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        with open(f"temp_{uuid.uuid4().hex}.pdf", "wb") as tmp_file:
            tmp_file.write(uploaded_file.read())
            pdf_path = tmp_file.name

            # Load the text
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()
            except Exception as e:
                st.error(f"Failed to load PDF: {e}")

            all_texts = [doc.page_content for doc in documents]

            # Chunking logic
            all_chunks = []
            if chunking_method == "Recursive":
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap), separators=["\n\n", "\n", " "]
                )
                for text in all_texts:
                    chunks = splitter.split_text(text)
                    all_chunks.extend(chunks)
            elif chunking_method == "Hybrid":
                hybrid_chunker = HybridChunker(chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
                for text in all_texts:
                    chunks = hybrid_chunker.process(text)
                    all_chunks.extend(chunks)

            st.write(f"Processed {len(all_chunks)} chunks using {chunking_method} chunking!")
            for i, chunk in enumerate(all_chunks[:10]):
                st.text_area(f"Chunk {i+1}", chunk, height=100)
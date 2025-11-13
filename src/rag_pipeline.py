"""
rag_pipeline.py
RAG pipeline using Groq Llama (ChatGroq) + Chroma + SentenceTransformers.
"""

# Disable HF tokenizer parallel warnings
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import List, Dict

from langchain_groq import ChatGroq
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
from pypdf import PdfReader

# ======================================================
# ENV + GROQ MODEL
# ======================================================
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found. Set it in .env")

# USE EXACT MODEL YOU PROVIDED
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    max_tokens=512
)

# ======================================================
# LOCAL EMBEDDINGS (Free)
# ======================================================
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_texts(texts: List[str]) -> List[List[float]]:
    arr = embedder.encode(texts, show_progress_bar=False)
    return [x.tolist() for x in arr]


# ======================================================
# CHROMA VECTOR STORE
# ======================================================
CHROMA_PATH = os.getenv("CHROMA_PATH", "./vector_data")
chroma_client = PersistentClient(path=CHROMA_PATH)

def get_collection(name="docs"):
    return chroma_client.get_or_create_collection(name)

def clear_collection(name="docs"):
    try:
        chroma_client.delete_collection(name)
    except Exception:
        pass


# ======================================================
# DOCUMENT LOADING (PDF / TXT / URL)
# ======================================================
def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    pages = [p.extract_text() or "" for p in reader.pages]
    return "\n".join(pages)

def read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def read_url(url: str) -> str:
    try:
        html = requests.get(url, timeout=10).text
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return " ".join(soup.get_text(separator=" ").split())
    except Exception as e:
        return f"__ERROR_LOADING_URL__: {e}"

def load_document(path_or_url: str) -> str:
    if path_or_url.startswith("http"):
        return read_url(path_or_url)

    ext = os.path.splitext(path_or_url)[1].lower()
    if ext == ".pdf":
        return read_pdf(path_or_url)
    if ext in [".txt", ".md"]:
        return read_txt(path_or_url)

    raise ValueError(f"Unsupported file type: {ext}")


# ======================================================
# CHUNKING
# ======================================================
def chunk_text(text: str, chunk_size=700, overlap=100) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
        i += chunk_size - overlap
    return chunks


# ======================================================
# ADD DOCUMENTS TO CHROMA
# ======================================================
def add_documents(path_or_url: str, clear_collection_first=True) -> int:
    if clear_collection_first:
        clear_collection("docs")

    text = load_document(path_or_url)
    if text.startswith("__ERROR_LOADING_URL__"):
        raise RuntimeError(text)

    chunks = chunk_text(text)
    embeddings = embed_texts(chunks)

    ids = [f"doc_{i}" for i in range(len(chunks))]
    metas = [{"source": path_or_url, "chunk_index": i} for i in range(len(chunks))]

    collection = get_collection()
    collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metas)

    return len(chunks)


# ======================================================
# RETRIEVAL (DEDUP)
# ======================================================
def retrieve(query: str, k=4) -> List[Dict]:
    q_emb = embed_texts([query])[0]
    col = get_collection()

    res = col.query(
        query_embeddings=[q_emb],
        n_results=k,
        include=["documents", "metadatas"]
    )

    docs = res["documents"][0]
    metas = res["metadatas"][0]

    seen = set()
    out = []

    for d, m in zip(docs, metas):
        if d.strip() and d not in seen:
            seen.add(d)
            out.append({"text": d, "metadata": m})

    return out


# ======================================================
# ANSWER GENERATION (ONE SENTENCE ONLY)
# ======================================================
def answer_query(query: str, k=4) -> Dict:
    docs = retrieve(query, k)

    if not docs:
        return {
            "answer": "The context does not provide this information.",
            "sources": [],
            "raw": ""
        }

    # prepare context for LLM
    context_lines = []
    for idx, d in enumerate(docs, start=1):
        src = d["metadata"].get("source", "unknown")
        context_lines.append(f"[{idx}] Source: {src}\n{d['text']}")

    context = "\n\n".join(context_lines)

    prompt = f"""
Use ONLY the following context to answer the question in ONE short sentence.
Do NOT repeat yourself.
If the answer is not in the context, say: "The context does not provide this information."

Context:
{context}

Question:
{query}

Answer (one sentence) + Sources:
"""

    resp = llm.invoke(prompt)
    full = getattr(resp, "content", str(resp)).strip()

    # Try to extract sources
    answer = full
    sources = []

    if "Sources:" in full:
        ans, srcs = full.rsplit("Sources:", 1)
        answer = ans.strip()

        import re
        sources = re.findall(r"\[\d+\]", srcs)

    return {
        "answer": answer,
        "sources": sources,
        "raw": full
    }

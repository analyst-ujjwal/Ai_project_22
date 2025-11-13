# RAG Groq Chatbot — Professional

A polished Retrieval-Augmented Generation (RAG) chatbot using Groq Llama models.

Features
- Groq Chat Llama (ChatGroq / langchain_groq) as the LLM (configurable via .env)
- Free local embeddings via `sentence-transformers` (no OpenAI credits)
- Persistent Chroma vector store
- PDF / TXT / Website ingestion
- Deduplication and overlap-tuned chunking to avoid repeated answers
- Streamlit frontend with source display and DB controls
- Lightweight, production-minded layout

---

## Quickstart

1. Clone or copy repository and change into `rag-groq-chatbot/src`:

```bash
cd rag-groq-chatbot

"""
Streamlit frontend for the professional RAG Groq Chatbot.
- imports add_documents, answer_query, clear_collection from rag_pipeline.py
- concise UI: upload files / enter URLs / chat / show sources
"""

import os
import tempfile
import time
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import add_documents, answer_query, clear_collection, get_collection, retrieve

load_dotenv()

st.set_page_config(page_title="RAG Groq Chatbot", layout="wide")
st.title("📚 RAG Groq Chatbot -  Project - 22")

# --- Sidebar controls
st.sidebar.markdown("## Knowledge Base Controls")
if "collection_name" not in st.session_state:
    st.session_state.collection_name = "docs"

uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT", type=["pdf", "txt"])
url_input = st.sidebar.text_input("Or paste a website URL (https://...)")
clear_db_button = st.sidebar.button("Clear knowledge base")

if clear_db_button:
    clear_collection(st.session_state.collection_name)
    st.sidebar.success("Knowledge base cleared.")
    time.sleep(0.2)

# File ingestion
if uploaded_file:
    suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    # Optionally clear previous data to avoid mixing KBs
    clear_collection(st.session_state.collection_name)
    num = add_documents(path, clear_collection_first=False)
    st.sidebar.success(f"Indexed {num} chunks from uploaded file.")

if url_input:
    clear_collection(st.session_state.collection_name)
    try:
        num = add_documents(url_input, clear_collection_first=False)
        st.sidebar.success(f"Indexed {num} chunks from URL.")
    except Exception as e:
        st.sidebar.error(str(e))

# --- Main chat layout
if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("Chat")
    user_query = st.chat_input("Ask about your ingested document(s)...")

    if user_query:
        st.session_state.history.append(("user", user_query))
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                res = answer_query(user_query, k=4)
                answer = res.get("answer", "No answer.")
                sources = res.get("sources", [])
                raw = res.get("raw", "")

                st.write(answer)
                if sources:
                    st.markdown("**Sources used:** " + " ".join(sources))
                else:
                    st.markdown("**Sources used:** None")

        st.session_state.history.append(("assistant", answer))

    # Conversation history
    st.markdown("---")
    st.markdown("### Conversation history")
    for role, msg in st.session_state.history:
        with st.chat_message(role):
            st.write(msg)

with col2:
    st.subheader("Knowledge base")

    try:
        coll = get_collection()  # from rag_pipeline

        # collection name
        name = getattr(coll, "name", "docs")
        st.markdown(f"**Collection:** `{name}`")

        # try to get a chunk count (Chroma has different versions; try safe fallbacks)
        count = "unknown"
        try:
            # modern Chroma collection may have .count()
            count = coll.count()
        except Exception:
            try:
                # fallback: coll.get() returns dict with lists
                all_data = coll.get(include=["ids"])
                if isinstance(all_data, dict):
                    ids = all_data.get("ids", [])
                    count = len(ids)
            except Exception:
                count = "unknown"

        st.markdown(f"**Indexed chunks:** {count}")

        # show a short preview of sources (metadatas)
        preview_limit = 8
        try:
            got = coll.get(include=["metadatas"], limit=preview_limit)
            metas_list = got.get("metadatas", [])
            # metas_list may be nested: list of lists (one result list)
            sources = []
            if metas_list:
                # handle both shapes: metas_list == [ {..}, {..}, ... ] OR metas_list == [[{..},...]]
                flat = metas_list[0] if isinstance(metas_list[0], list) else metas_list
                for m in flat:
                    if not isinstance(m, dict):
                        continue
                    src = m.get("source") or m.get("uri") or m.get("path")
                    if src and src not in sources:
                        sources.append(src)
                    if len(sources) >= preview_limit:
                        break

            if sources:
                st.markdown("**Preview sources:**")
                for s in sources:
                    st.write(f"- {s}")
            else:
                st.info("No sources indexed yet. Upload a file or paste a URL in the sidebar.")
        except Exception:
            st.info("No source preview available for this Chroma version.")

        st.markdown("---")
        st.markdown("You can upload a new file or paste a URL in the sidebar.")
        st.markdown("Click **Clear knowledge base** in the sidebar to reset the KB before adding another document.")
    except Exception as e:
        st.error(f"Could not read knowledge base: {e}")

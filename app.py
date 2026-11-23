import os
import streamlit as st
from rag_engine import HybridRAGEngine

st.set_page_config(page_title="Hybrid RAG: Spotify Architecture", layout="wide")
st.title("Hybrid RAG: Vector DB (Qdrant) + Knowledge Graph (Neo4j)")

@st.cache_resource
def get_engine():
    return HybridRAGEngine()

engine = get_engine()

# Sidebar: PDF Ingestion
with st.sidebar:
    st.header("Document Ingestion")
    uploaded_file = st.file_uploader("Upload Architecture PDF", type=["pdf"])

    if st.button("Ingest into Vector DB & Knowledge Graph"):
        if uploaded_file:
            save_path = os.path.join("/app/data", uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            with st.spinner("Chunking, embedding, and extracting Knowledge Graph..."):
                chunks, g_docs = engine.ingest_pdf(save_path)
            st.success(f"Ingested {chunks} vector chunks and {g_docs} graph documents!")
        else:
            st.warning("Please upload a PDF first.")

# Main Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about microservices, databases, routes, or features..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving from Qdrant & Neo4j..."):
            result = engine.query(query)
        st.markdown(result["answer"])

        with st.expander("Inspect Hybrid Retrieval Sources"):
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Knowledge Graph (Neo4j)")
                st.code(result["graph_context"], language="text")
            with col2:
                st.subheader("Vector Chunks (Qdrant)")
                st.text(result["vector_context"])

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})
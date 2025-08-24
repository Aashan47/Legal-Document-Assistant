"""
Streamlit UI for the Legal Document Analyzer.
"""
import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any
import time
from pathlib import Path

from src.core.config import settings
from src.utils.logging import app_logger


# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API base URL
API_BASE_URL = f"http://{settings.api_host}:{settings.api_port}"


def check_api_health() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def upload_documents(files: List) -> Dict[str, Any]:
    """Upload documents to the API."""
    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file, "application/pdf")))
        
        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files_data,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        st.error(f"Error uploading documents: {str(e)}")
        return {}


def query_documents(question: str, top_k: int = 5) -> Dict[str, Any]:
    """Query the document knowledge base."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/query",
            json={"question": question, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        st.error(f"Error querying documents: {str(e)}")
        return {}


def get_stats() -> Dict[str, Any]:
    """Get knowledge base statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
        
    except Exception as e:
        st.error(f"Error getting stats: {str(e)}")
        return {}


def get_recent_queries() -> List[Dict[str, Any]]:
    """Get recent query history."""
    try:
        response = requests.get(f"{API_BASE_URL}/queries/recent", timeout=10)
        response.raise_for_status()
        return response.json().get("queries", [])
        
    except Exception as e:
        st.error(f"Error getting query history: {str(e)}")
        return []


def clear_knowledge_base():
    """Clear the knowledge base."""
    try:
        response = requests.delete(f"{API_BASE_URL}/clear", timeout=10)
        response.raise_for_status()
        return True
        
    except Exception as e:
        st.error(f"Error clearing knowledge base: {str(e)}")
        return False


def main():
    """Main Streamlit application."""
    
    # Title and description
    st.title("⚖️ Legal Document Analyzer")
    st.markdown("*AI-powered analysis of legal contracts using RAG and LLMs*")
    
    # Check API health
    api_healthy = check_api_health()
    if not api_healthy:
        st.error("🚨 API server is not running. Please start the FastAPI server first.")
        st.markdown(f"Expected API at: `{API_BASE_URL}`")
        st.code("python -m src.api.main", language="bash")
        return
    
    st.success("✅ API server is running")
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Control Panel")
        
        # Statistics
        st.subheader("📊 Knowledge Base Stats")
        stats = get_stats()
        if stats:
            st.metric("Documents", stats.get("total_documents", 0))
            st.metric("Vector Dimension", stats.get("embedding_dimension", 0))
            st.metric("Queries Made", stats.get("query_count", 0))
            if stats.get("query_count", 0) > 0:
                confidence = stats.get("average_confidence", 0)
                st.metric("Avg Confidence", f"{confidence:.2f}")
        
        # Clear knowledge base
        st.subheader("🗑️ Management")
        if st.button("Clear Knowledge Base", type="secondary"):
            if clear_knowledge_base():
                st.success("Knowledge base cleared!")
                st.rerun()
        
        # Configuration
        st.subheader("⚙️ Configuration")
        st.write(f"**Model**: {settings.default_llm}")
        st.write(f"**Chunk Size**: {settings.chunk_size}")
        st.write(f"**Temperature**: {settings.llm_temperature}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "🔍 Query Documents", "📈 Query History"])
    
    with tab1:
        st.header("📄 Upload Legal Documents")
        st.markdown("Upload PDF legal documents (contracts, NDAs, agreements, etc.) to build your knowledge base.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            st.write(f"Selected {len(uploaded_files)} file(s):")
            for file in uploaded_files:
                st.write(f"- {file.name} ({file.size:,} bytes)")
            
            if st.button("📤 Upload & Process Documents", type="primary"):
                with st.spinner("Uploading and processing documents..."):
                    result = upload_documents(uploaded_files)
                    
                    if result:
                        st.success(f"✅ {result.get('message', 'Upload completed')}")
                        
                        # Show processing details if available
                        files_info = result.get('files', [])
                        if files_info:
                            st.write("**Uploaded files:**")
                            for filename in files_info:
                                st.write(f"- {filename}")
                        
                        st.info("📋 Documents are being processed in the background. You can start querying in a few moments.")
                        
                        # Auto-refresh stats after upload
                        time.sleep(2)
                        st.rerun()
    
    with tab2:
        st.header("🔍 Query Legal Documents")
        st.markdown("Ask natural language questions about your uploaded legal documents.")
        
        # Query interface
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_area(
                "Ask a question about your legal documents:",
                placeholder="e.g., What are the termination clauses in the contract? What are the liability limitations?",
                height=100
            )
        with col2:
            top_k = st.selectbox("Results to consider", [3, 5, 10, 15], index=1)
        
        if st.button("🔍 Ask Question", type="primary", disabled=not query.strip()):
            if query.strip():
                with st.spinner("Analyzing documents..."):
                    result = query_documents(query, top_k)
                    
                    if result:
                        # Display answer
                        st.subheader("💬 Answer")
                        st.write(result.get("answer", "No answer generated"))
                        
                        # Display confidence
                        confidence = result.get("confidence", 0)
                        confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                        st.markdown(f"**Confidence:** :{confidence_color}[{confidence:.2f}]")
                        
                        # Warning for low confidence
                        if confidence < settings.confidence_threshold:
                            st.warning("⚠️ Low confidence answer. Consider uploading more relevant documents or rephrasing your question.")
                        
                        # Display sources
                        sources = result.get("sources", [])
                        if sources:
                            st.subheader("📑 Sources")
                            for i, source in enumerate(sources):
                                with st.expander(f"Source {i+1} (Score: {source['score']:.3f})"):
                                    st.write(source["content"])
                                    if source.get("metadata"):
                                        st.json(source["metadata"])
                        
                        # Model info
                        model_info = result.get("model_info", {})
                        if model_info:
                            with st.expander("🤖 Model Information"):
                                st.json(model_info)
        
        # Example queries
        st.subheader("💡 Example Questions")
        example_queries = [
            "What are the termination clauses in this contract?",
            "What are the liability limitations mentioned?",
            "What are the payment terms and conditions?",
            "Are there any confidentiality obligations?",
            "What are the governing law and jurisdiction clauses?",
            "What are the intellectual property provisions?",
            "Are there any penalty clauses for breach of contract?",
            "What is the duration or term of this agreement?"
        ]
        
        for example in example_queries:
            if st.button(f"📝 {example}", key=f"example_{hash(example)}"):
                st.session_state.example_query = example
                st.rerun()
        
        # Auto-fill example query
        if hasattr(st.session_state, 'example_query'):
            st.text_area(
                "Selected example:",
                value=st.session_state.example_query,
                height=50,
                disabled=True
            )
    
    with tab3:
        st.header("📈 Query History")
        st.markdown("Review recent queries and their confidence scores.")
        
        queries = get_recent_queries()
        
        if queries:
            for i, query_data in enumerate(reversed(queries)):
                with st.expander(f"Query {len(queries)-i}: {query_data['question'][:100]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Question:**")
                        st.write(query_data["question"])
                        st.write("**Answer:**")
                        st.write(query_data["answer"])
                    
                    with col2:
                        st.metric("Confidence", f"{query_data['confidence']:.2f}")
                        st.metric("Sources Used", query_data["sources_count"])
                        st.write(f"**Model:** {query_data['model']}")
        else:
            st.info("No queries yet. Upload documents and start asking questions!")


if __name__ == "__main__":
    main()

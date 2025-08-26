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
from src.models.model_config import AVAILABLE_MODELS, get_recommended_model, ModelConfig


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


def query_documents(question: str, top_k: int = 5, model_name: str = None) -> Dict[str, Any]:
    """Query the document knowledge base with optional model selection."""
    try:
        payload = {"question": question, "top_k": top_k}
        if model_name:
            payload["model_name"] = model_name
            
        response = requests.post(
            f"{API_BASE_URL}/query",
            json=payload,
            timeout=120  # Increased timeout for complex legal analysis
        )
        response.raise_for_status()
        return response.json()
        
    except requests.exceptions.Timeout:
        st.error("⏳ Request timed out. The model is processing a complex legal analysis. Please try a shorter question or wait for the system to optimize.")
        return {}
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
    
    # Initialize session state for chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "api_healthy" not in st.session_state:
        st.session_state.api_healthy = False
    
    # Title and description
    st.title("⚖️ Legal Document Analyzer")
    st.markdown("*AI-powered analysis of legal contracts using RAG and LLMs*")
    
    # Check API health
    api_healthy = check_api_health()
    st.session_state.api_healthy = api_healthy
    
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
            st.metric("Chat Messages", len(st.session_state.chat_history))
            if stats.get("query_count", 0) > 0:
                confidence = stats.get("average_confidence", 0)
                st.metric("Avg Confidence", f"{confidence:.2f}")
        
        # Clear options
        st.subheader("🗑️ Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("Clear DB", type="secondary"):
                if clear_knowledge_base():
                    st.success("Knowledge base cleared!")
                    st.rerun()
        
        # Configuration
        st.subheader("⚙️ Configuration")
        
        # Model Selection
        st.markdown("**🤖 AI Model Selection**")
        
        # Initialize selected model in session state
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = "flan-t5-large"  # Use the key, not the model_id
        
        # Create model options for selectbox
        model_options = {}
        for model_key, config in AVAILABLE_MODELS.items():
            # Create a descriptive label
            label = f"{config.name} - {config.description[:50]}..."
            model_options[label] = model_key  # Use the key, not model_id
        
        # Model selection dropdown
        selected_label = st.selectbox(
            "Choose AI Model:",
            options=list(model_options.keys()),
            index=list(model_options.values()).index(st.session_state.selected_model),
            help="Select the AI model for analyzing your legal documents"
        )
        
        # Update session state
        st.session_state.selected_model = model_options[selected_label]
        
        # Show model details
        selected_config = AVAILABLE_MODELS[st.session_state.selected_model]
        st.markdown(f"**Strengths**: {selected_config.strengths}")
        st.markdown(f"**Best for**: {selected_config.best_for}")
        
        # Show performance metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Speed", f"{selected_config.speed}/5", help="Response speed rating")
        with col2:
            st.metric("Quality", f"{selected_config.quality}/5", help="Response quality rating")
        with col3:
            st.metric("Size", f"{selected_config.size}/5", help="Model size (higher = more resource intensive)")
        
        st.divider()
        
        # Model Cache Status
        st.markdown("**🚀 Model Cache Status**")
        try:
            from src.models.model_manager import get_model_manager
            manager = get_model_manager()
            cache_info = manager.get_cache_info()
            
            # Show cache status
            cached_count = cache_info['total_cached']
            total_count = cache_info['total_available']
            
            if cached_count == total_count:
                st.success(f"✅ All {total_count} models cached - Instant switching enabled!")
            elif cached_count > 0:
                st.warning(f"⚠️ {cached_count}/{total_count} models cached")
            else:
                st.error("❌ No models cached - First load will be slower")
            
            if cached_count > 0:
                st.write(f"💾 Cache size: {cache_info['total_size_gb']:.1f} GB")
                
            # Show preload button if not all cached
            if cached_count < total_count:
                st.info("💡 Run `python preload_models.py` to cache all models for instant switching")
                
        except Exception as e:
            st.warning("⚠️ Cache status unavailable")
        
        st.divider()
        
        # Other configuration info
        st.markdown("**System Settings**")
        st.write(f"**LLM Type**: {settings.default_llm}")
        st.write(f"**Chunk Size**: {settings.chunk_size}")
        st.write(f"**Temperature**: {settings.llm_temperature}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "� Chat with Documents", "📈 Query History"])
    
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
        st.header("� Chat with Legal Documents")
        st.markdown("Ask questions about your uploaded legal documents. Your chat history is maintained throughout the session.")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for i, message in enumerate(st.session_state.chat_history):
                    if message["role"] == "user":
                        with st.chat_message("user"):
                            st.write(f"**Q{i//2 + 1}:** {message['content']}")
                    else:
                        with st.chat_message("assistant"):
                            st.write(f"**A{i//2 + 1}:** {message['content']}")
                            
                            # Show confidence and sources if available
                            if "confidence" in message:
                                confidence = message["confidence"]
                                confidence_color = "green" if confidence > 0.7 else "orange" if confidence > 0.4 else "red"
                                st.markdown(f"*Confidence: :{confidence_color}[{confidence:.1%}]*")
                            
                            if "sources" in message and message["sources"]:
                                with st.expander(f"📑 View {len(message['sources'])} sources"):
                                    for j, source in enumerate(message["sources"], 1):
                                        st.write(f"**Source {j}** (Score: {source['score']:.3f})")
                                        st.write(source["content"][:200] + "..." if len(source["content"]) > 200 else source["content"])
                                        st.write("---")
                            
                            # Show debug info for troubleshooting
                            if "processing_time" in message:
                                st.caption(f"⏱️ Processing time: {message['processing_time']:.1f}s")
            else:
                st.info("💬 Start your conversation by asking a question about your legal documents below.")
                
                # Show current database status
                stats = get_stats()
                if stats and stats.get("total_documents", 0) > 0:
                    st.success(f"📊 Knowledge base ready with {stats['total_documents']} documents")
                else:
                    st.warning("📋 No documents in knowledge base. Please upload documents first!")
        
        # Chat input section
        st.markdown("---")
        
        # Query interface
        col1, col2 = st.columns([4, 1])
        with col1:
            query = st.text_area(
                "Ask a question about your legal documents:",
                placeholder="e.g., What are the termination clauses in the contract? What are the liability limitations?",
                height=100,
                key="query_input"
            )
        with col2:
            top_k = st.selectbox("Results to consider", [3, 5, 10, 15], index=1)
            st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
        
        # Check for pending query from example buttons
        pending_query = st.session_state.get("pending_query", "")
        actual_query = pending_query if pending_query else query
        
        # Ask button and processing
        if st.button("🔍 Ask Question", type="primary", disabled=not actual_query.strip()) or pending_query:
            if actual_query.strip():
                # Clear pending query if it was used
                if pending_query:
                    st.session_state.pending_query = ""
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": actual_query.strip()
                })
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Searching relevant documents...")
                progress_bar.progress(25)
                
                with st.spinner("Analyzing documents and generating detailed response..."):
                    status_text.text("🤖 AI model analyzing legal content...")
                    progress_bar.progress(50)
                    
                    start_time = time.time()
                    # Get the actual model_id from the selected model key
                    selected_model_id = AVAILABLE_MODELS[st.session_state.selected_model].model_id
                    result = query_documents(actual_query.strip(), top_k, selected_model_id)
                    end_time = time.time()
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ Analysis complete! ({end_time - start_time:.1f}s)")
                    
                    if result:
                        # Add assistant response to chat history
                        assistant_message = {
                            "role": "assistant",
                            "content": result.get("answer", "No answer generated"),
                            "confidence": result.get("confidence", 0),
                            "sources": result.get("sources", []),
                            "model_info": result.get("model_info", {}),
                            "processing_time": end_time - start_time
                        }
                        st.session_state.chat_history.append(assistant_message)
                        
                        # Warning for low confidence
                        confidence = result.get("confidence", 0)
                        if confidence < settings.confidence_threshold:
                            st.warning("⚠️ Low confidence answer. Consider uploading more relevant documents or rephrasing your question.")
                    else:
                        # Add error message
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": "❌ Sorry, I couldn't process your question. Please try again.",
                            "confidence": 0,
                            "sources": []
                        })
                    
                    # Clear progress indicators and rerun to show updated chat
                    progress_bar.empty()
                    status_text.empty()
                    
                    # Rerun to show updated chat
                    st.rerun()
        
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
        
        cols = st.columns(2)
        for i, example in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(f"📝 {example[:40]}...", key=f"example_{hash(example)}", help=example):
                    # Store the example query in session state and trigger processing
                    st.session_state.pending_query = example
                    st.rerun()
    
    with tab3:
        st.header("📈 Query History & Analytics")
        st.markdown("Review recent queries and their confidence scores.")
        
        # Current session history
        if st.session_state.chat_history:
            st.subheader("📊 Current Session Analytics")
            
            # Calculate session stats
            user_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
            assistant_messages = [msg for msg in st.session_state.chat_history if msg["role"] == "assistant" and "confidence" in msg]
            
            if assistant_messages:
                avg_confidence = sum(msg["confidence"] for msg in assistant_messages) / len(assistant_messages)
                avg_processing_time = sum(msg.get("processing_time", 0) for msg in assistant_messages) / len(assistant_messages)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Questions Asked", len(user_messages))
                with col2:
                    st.metric("Avg Confidence", f"{avg_confidence:.1%}")
                with col3:
                    st.metric("Avg Response Time", f"{avg_processing_time:.1f}s")
            
            st.subheader("� Current Session Chat History")
            for i in range(0, len(st.session_state.chat_history), 2):
                if i + 1 < len(st.session_state.chat_history):
                    user_msg = st.session_state.chat_history[i]
                    assistant_msg = st.session_state.chat_history[i + 1]
                    
                    with st.expander(f"Q{i//2 + 1}: {user_msg['content'][:80]}..."):
                        st.write("**Question:**")
                        st.write(user_msg['content'])
                        st.write("**Answer:**")
                        st.write(assistant_msg['content'])
                        
                        if "confidence" in assistant_msg:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Confidence", f"{assistant_msg['confidence']:.1%}")
                            with col2:
                                st.metric("Sources", len(assistant_msg.get('sources', [])))
        
        # API-level query history
        st.subheader("🌐 Recent API Queries")
        queries = get_recent_queries()
        
        if queries:
            for i, query_data in enumerate(reversed(queries[-10:])):  # Show last 10
                with st.expander(f"API Query {len(queries)-i}: {query_data['question'][:80]}..."):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Question:**")
                        st.write(query_data["question"])
                        st.write("**Answer:**")
                        st.write(query_data["answer"][:300] + "..." if len(query_data["answer"]) > 300 else query_data["answer"])
                    
                    with col2:
                        st.metric("Confidence", f"{query_data['confidence']:.2f}")
                        st.metric("Sources Used", query_data["sources_count"])
                        st.write(f"**Model:** {query_data['model']}")
        else:
            st.info("No API queries yet. Upload documents and start asking questions!")


if __name__ == "__main__":
    main()

"""
Legal Document Analyzer - Streamlit Cloud Version
This is the main entry point for Streamlit Cloud deployment.
"""
import streamlit as st
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import time

# Add src to Python path
sys.path.append(str(Path(__file__).parent / "src"))

# Import core modules with fallbacks
try:
    from core.vector_db import VectorDatabase
    from core.rag import RAGPipeline
    from data.processor import DocumentProcessor
    from models.model_config import AVAILABLE_MODELS
    from utils.logging import app_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    # Create simple logger fallback
    import logging
    app_logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f4e79;
    }
    .user-message {
        background-color: #e8f4f8;
        border-left-color: #1f4e79;
    }
    .assistant-message {
        background-color: #f0f8f0;
        border-left-color: #2d7d32;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def initialize_system():
    """Initialize the RAG system (cached for performance)."""
    try:
        # Create data directories
        data_dir = Path("./data")
        data_dir.mkdir(exist_ok=True)
        (data_dir / "uploads").mkdir(exist_ok=True)
        (data_dir / "vector_db").mkdir(exist_ok=True)
        
        # Initialize vector database
        vector_db = VectorDatabase()
        
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        app_logger.info("System initialized successfully for Streamlit Cloud")
        return vector_db, rag_pipeline, True
        
    except Exception as e:
        app_logger.error(f"Failed to initialize system: {e}")
        st.error(f"System initialization failed: {e}")
        return None, None, False

def upload_and_process_documents(uploaded_files: List, vector_db: VectorDatabase) -> Dict[str, Any]:
    """Process uploaded documents and add to vector database."""
    try:
        processor = DocumentProcessor()
        results = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, uploaded_file in enumerate(uploaded_files):
            status_text.text(f"Processing {uploaded_file.name}...")
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file.flush()
                
                # Process the document
                documents = processor.process_file(tmp_file.name)
                
                # Add to vector database
                for doc in documents:
                    vector_db.add_document(doc["content"], {
                        "filename": uploaded_file.name,
                        "size": uploaded_file.size,
                        "page": doc.get("page", 0),
                        "chunk_id": doc.get("chunk_id", i)
                    })
                
                results.append({
                    "filename": uploaded_file.name,
                    "chunks": len(documents)
                })
                
                # Clean up temp file
                os.unlink(tmp_file.name)
            
            # Update progress
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        progress_bar.empty()
        status_text.empty()
        
        return {
            "success": True,
            "message": f"Successfully processed {len(results)} documents",
            "files": results
        }
        
    except Exception as e:
        app_logger.error(f"Error processing documents: {e}")
        return {
            "success": False,
            "error": str(e)
        }

def query_documents(question: str, rag_pipeline: RAGPipeline, model_name: str = None, top_k: int = 5) -> Dict[str, Any]:
    """Query documents using the RAG pipeline."""
    try:
        # Use only the model_id for the RAG pipeline
        if model_name and model_name in AVAILABLE_MODELS:
            model_id = AVAILABLE_MODELS[model_name].model_id
        else:
            model_id = None
            
        result = rag_pipeline.query(question, top_k=top_k, model_name=model_id)
        return result
        
    except Exception as e:
        app_logger.error(f"Error querying documents: {e}")
        return {
            "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
            "confidence": 0.0,
            "sources": [],
            "error": str(e)
        }

def display_chat_message(message: Dict[str, Any], is_user: bool = False):
    """Display a chat message with proper styling."""
    if is_user:
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {message["content"]}</div>', unsafe_allow_html=True)
    else:
        confidence = message.get("confidence", 0)
        confidence_color = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.4 else "🔴"
        
        st.markdown(f'<div class="chat-message assistant-message">', unsafe_allow_html=True)
        st.markdown(f"**🤖 AI Assistant:** {message['content']}")
        
        # Show confidence and sources in expander
        with st.expander(f"{confidence_color} Confidence: {confidence:.2f} | Sources: {len(message.get('sources', []))}"):
            if message.get('sources'):
                st.markdown("**Sources:**")
                for i, source in enumerate(message['sources'][:3]):  # Show top 3 sources
                    st.markdown(f"**Source {i+1}** (Score: {source.get('score', 0):.2f})")
                    st.text(source.get('content', '')[:200] + "...")
        
        st.markdown('</div>', unsafe_allow_html=True)

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("*AI-powered analysis of your legal contracts and documents*")
    
    # Initialize system
    with st.spinner("🚀 Initializing AI system..."):
        vector_db, rag_pipeline, init_success = initialize_system()
    
    if not init_success:
        st.error("❌ Failed to initialize the AI system. Please refresh the page.")
        st.stop()
    
    st.success("✅ AI system ready!")
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "flan-t5-large"
    if "documents_uploaded" not in st.session_state:
        st.session_state.documents_uploaded = 0
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Control Panel")
        
        # Model selection (only show available models)
        available_models = list(AVAILABLE_MODELS.keys())
        if available_models:
            st.subheader("🤖 AI Model Selection")
            
            # Create model options
            model_options = {}
            for model_key in available_models:
                config = AVAILABLE_MODELS[model_key]
                label = f"{config.name} - {config.description[:40]}..."
                model_options[label] = model_key
            
            selected_label = st.selectbox(
                "Choose AI Model:",
                options=list(model_options.keys()),
                index=list(model_options.values()).index(st.session_state.selected_model) if st.session_state.selected_model in model_options.values() else 0,
                help="Select the AI model for analyzing your legal documents"
            )
            
            st.session_state.selected_model = model_options[selected_label]
            
            # Show model details
            selected_config = AVAILABLE_MODELS[st.session_state.selected_model]
            st.info(f"**{selected_config.name}**\n\n{selected_config.strengths}")
            
            # Performance metrics
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Speed", selected_config.speed)
            with col2:
                st.metric("Quality", selected_config.quality)
        
        st.divider()
        
        # Statistics
        st.subheader("📊 Knowledge Base Stats")
        try:
            stats = vector_db.get_stats()
            st.metric("Documents", stats.get("total_documents", 0))
            st.metric("Vector Dimension", stats.get("embedding_dimension", 0))
            st.metric("Chat Messages", len(st.session_state.chat_history))
        except Exception as e:
            st.error(f"Error loading stats: {e}")
        
        # Management options
        st.subheader("🗑️ Management")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Chat", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
        with col2:
            if st.button("Clear DB", type="secondary", help="Clear all uploaded documents"):
                try:
                    vector_db = VectorDatabase()  # Reinitialize
                    st.session_state.documents_uploaded = 0
                    st.success("Knowledge base cleared!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error clearing database: {e}")
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Chat with Documents", "📈 Query History"])
    
    with tab1:
        st.header("📄 Upload Legal Documents")
        st.markdown("Upload PDF legal documents to build your knowledge base.")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Select one or more PDF files to upload"
        )
        
        if uploaded_files:
            if st.button("📤 Process Documents", type="primary"):
                with st.spinner("Processing documents..."):
                    result = upload_and_process_documents(uploaded_files, vector_db)
                
                if result.get("success"):
                    st.success(result["message"])
                    st.session_state.documents_uploaded += len(uploaded_files)
                    
                    # Show processed files
                    for file_info in result["files"]:
                        st.write(f"✅ {file_info['filename']} - {file_info['chunks']} chunks")
                else:
                    st.error(f"Processing failed: {result.get('error', 'Unknown error')}")
    
    with tab2:
        st.header("💬 Chat with Your Documents")
        
        if st.session_state.documents_uploaded == 0:
            st.warning("⚠️ Please upload some documents first in the 'Document Upload' tab.")
        else:
            # Display chat history
            if st.session_state.chat_history:
                st.subheader("📜 Conversation History")
                for message in st.session_state.chat_history:
                    if message["role"] == "user":
                        display_chat_message(message, is_user=True)
                    else:
                        display_chat_message(message, is_user=False)
            
            # Query interface
            st.subheader("🔍 Ask a Question")
            
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_area(
                    "Ask about your documents:",
                    placeholder="e.g., What are the termination clauses? What are the liability limitations?",
                    height=100
                )
            with col2:
                top_k = st.selectbox("Results to consider", [3, 5, 10], index=1)
                st.markdown("<br>", unsafe_allow_html=True)
            
            if st.button("🔍 Ask Question", type="primary", disabled=not query.strip()):
                if query.strip():
                    # Add user message to chat history
                    st.session_state.chat_history.append({
                        "role": "user",
                        "content": query.strip()
                    })
                    
                    # Process query
                    with st.spinner("🤖 AI is analyzing your documents..."):
                        start_time = time.time()
                        result = query_documents(
                            query.strip(), 
                            rag_pipeline, 
                            st.session_state.selected_model, 
                            top_k
                        )
                        end_time = time.time()
                    
                    # Add assistant response to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": result.get("answer", "No answer generated"),
                        "confidence": result.get("confidence", 0),
                        "sources": result.get("sources", []),
                        "processing_time": end_time - start_time
                    }
                    st.session_state.chat_history.append(assistant_message)
                    
                    # Show confidence warning if needed
                    if result.get("confidence", 0) < 0.5:
                        st.warning("⚠️ Low confidence answer. Consider uploading more relevant documents.")
                    
                    st.rerun()
            
            # Example queries
            st.subheader("💡 Example Questions")
            example_queries = [
                "What are the termination clauses in this contract?",
                "What are the liability limitations?",
                "What are the intellectual property provisions?",
                "What are the payment terms and conditions?"
            ]
            
            cols = st.columns(2)
            for i, example in enumerate(example_queries):
                with cols[i % 2]:
                    if st.button(f"📝 {example[:30]}...", key=f"example_{i}", help=example):
                        # Add to chat and process
                        st.session_state.chat_history.append({
                            "role": "user",
                            "content": example
                        })
                        
                        with st.spinner("🤖 Processing example query..."):
                            result = query_documents(example, rag_pipeline, st.session_state.selected_model)
                        
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result.get("answer", "No answer generated"),
                            "confidence": result.get("confidence", 0),
                            "sources": result.get("sources", [])
                        })
                        
                        st.rerun()
    
    with tab3:
        st.header("📈 Query History & Analytics")
        
        if st.session_state.chat_history:
            # Filter only user queries
            user_queries = [msg for msg in st.session_state.chat_history if msg["role"] == "user"]
            assistant_responses = [msg for msg in st.session_state.chat_history if msg["role"] == "assistant"]
            
            st.metric("Total Queries", len(user_queries))
            
            if assistant_responses:
                avg_confidence = sum(msg.get("confidence", 0) for msg in assistant_responses) / len(assistant_responses)
                st.metric("Average Confidence", f"{avg_confidence:.2f}")
            
            # Recent queries
            st.subheader("🔍 Recent Queries")
            for i, query in enumerate(reversed(user_queries[-10:])):  # Last 10 queries
                st.write(f"{len(user_queries) - i}. {query['content']}")
        else:
            st.info("No queries yet. Start chatting with your documents!")
    
    # Footer
    st.markdown("---")
    st.markdown("*Legal Document Analyzer - Powered by AI*")

if __name__ == "__main__":
    main()

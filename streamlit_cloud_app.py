"""
Streamlit Cloud Deployable Version - Legal Document Analyzer
This version combines all functionality into a single Streamlit app.
"""
import streamlit as st
import tempfile
import os
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Cloud-optimized imports
try:
    from core.vector_db import VectorDatabase
    from core.rag import RAGPipeline
    from data.processor import DocumentProcessor
    from models.model_config import AVAILABLE_MODELS
    from utils.logging import app_logger
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cloud-safe directory setup
@st.cache_data
def setup_directories():
    """Setup directories for cloud deployment."""
    base_dir = Path("./data")
    base_dir.mkdir(exist_ok=True)
    (base_dir / "uploads").mkdir(exist_ok=True)
    (base_dir / "vector_db").mkdir(exist_ok=True)
    return base_dir

@st.cache_resource
def initialize_system():
    """Initialize the system components."""
    try:
        # Setup directories
        setup_directories()
        
        # Initialize components
        vector_db = VectorDatabase()
        rag_pipeline = RAGPipeline()
        doc_processor = DocumentProcessor()
        
        app_logger.info("System initialized for cloud deployment")
        return vector_db, rag_pipeline, doc_processor, True
        
    except Exception as e:
        st.error(f"Failed to initialize system: {e}")
        return None, None, None, False

def display_model_selector():
    """Display model selection in sidebar."""
    st.sidebar.header("🤖 AI Model Selection")
    
    # Filter only working models (T5-based)
    working_models = {
        k: v for k, v in AVAILABLE_MODELS.items() 
        if "flan-t5" in k.lower() and k in AVAILABLE_MODELS
    }
    
    if not working_models:
        st.sidebar.error("No working models available")
        return "flan-t5-large"
    
    # Model selection
    model_names = list(working_models.keys())
    model_labels = [f"{working_models[k].name} - {working_models[k].description[:40]}..." for k in model_names]
    
    selected_idx = st.sidebar.selectbox(
        "Choose Model:",
        range(len(model_names)),
        format_func=lambda x: model_labels[x],
        index=0
    )
    
    selected_model = model_names[selected_idx]
    config = working_models[selected_model]
    
    # Display model info
    st.sidebar.info(f"""
    **{config.name}**
    
    **Strengths**: {config.strengths}
    
    **Best for**: {config.best_for}
    
    **Speed**: {config.speed} | **Quality**: {config.quality}
    """)
    
    return config.model_id

def upload_and_process_documents(doc_processor, vector_db):
    """Handle document upload and processing."""
    st.header("📄 Upload Legal Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload PDF legal documents for analysis"
    )
    
    if uploaded_files:
        if st.button("📤 Process Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            processed_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file.flush()
                        
                        # Process document
                        documents = doc_processor.process_file(tmp_file.name)
                        
                        # Add to vector database
                        for doc in documents:
                            vector_db.add_document(
                                doc['content'], 
                                {**doc['metadata'], 'filename': uploaded_file.name}
                            )
                        
                        processed_count += 1
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                        
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            status_text.text("✅ Processing complete!")
            st.success(f"Successfully processed {processed_count} documents!")
            st.rerun()

def query_interface(rag_pipeline, selected_model):
    """Handle document querying."""
    st.header("🔍 Ask Questions About Your Documents")
    
    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Display chat history
    if st.session_state.chat_history:
        st.subheader("💬 Conversation History")
        for i, message in enumerate(st.session_state.chat_history):
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI:** {message['content']}")
                if "confidence" in message:
                    conf_color = "green" if message["confidence"] > 0.7 else "orange" if message["confidence"] > 0.4 else "red"
                    st.markdown(f"<small style='color: {conf_color}'>Confidence: {message['confidence']:.2f}</small>", unsafe_allow_html=True)
                st.divider()
    
    # Query input
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_query = st.text_area(
            "Your Question:",
            placeholder="e.g., What are the termination clauses? What are the liability limitations?",
            height=100
        )
    
    with col2:
        top_k = st.selectbox("Results to analyze", [3, 5, 10], index=1)
        st.markdown("<br>", unsafe_allow_html=True)
    
    # Process query
    if st.button("🔍 Ask Question", type="primary", disabled=not user_query.strip()):
        if user_query.strip():
            # Add user message
            st.session_state.chat_history.append({
                "role": "user", 
                "content": user_query.strip()
            })
            
            # Process query
            with st.spinner("🤖 AI analyzing your documents..."):
                try:
                    result = rag_pipeline.query(
                        user_query.strip(), 
                        top_k=top_k,
                        model_name=selected_model
                    )
                    
                    # Add AI response
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": result.get("answer", "Sorry, I couldn't generate an answer."),
                        "confidence": result.get("confidence", 0.0),
                        "sources": result.get("sources", [])
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": "Sorry, I encountered an error processing your question.",
                        "confidence": 0.0
                    })
            
            st.rerun()
    
    # Example queries
    if not st.session_state.chat_history:
        st.subheader("💡 Example Questions")
        examples = [
            "What are the termination clauses in the contract?",
            "What are the liability limitations?",
            "What are the intellectual property provisions?",
            "What are the payment terms?"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}"):
                    st.session_state.pending_query = example
                    st.rerun()

def display_statistics(vector_db):
    """Display system statistics."""
    st.sidebar.header("📊 System Status")
    
    try:
        stats = vector_db.get_stats()
        st.sidebar.metric("Documents", stats.get("total_documents", 0))
        st.sidebar.metric("Embeddings", stats.get("embedding_dimension", "N/A"))
        
        if "chat_history" in st.session_state:
            st.sidebar.metric("Chat Messages", len(st.session_state.chat_history))
    
    except Exception as e:
        st.sidebar.error(f"Stats unavailable: {e}")
    
    # Clear options
    st.sidebar.header("🗑️ Management")
    if st.sidebar.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

def main():
    """Main application."""
    st.title("⚖️ Legal Document Analyzer")
    st.markdown("*AI-powered analysis of legal documents - Cloud Edition*")
    
    # Initialize system
    with st.spinner("🚀 Initializing AI system..."):
        vector_db, rag_pipeline, doc_processor, success = initialize_system()
    
    if not success:
        st.error("❌ Failed to initialize system. Please refresh the page.")
        return
    
    # Sidebar
    selected_model = display_model_selector()
    display_statistics(vector_db)
    
    # Main content tabs
    tab1, tab2 = st.tabs(["📄 Document Upload", "🔍 Ask Questions"])
    
    with tab1:
        upload_and_process_documents(doc_processor, vector_db)
    
    with tab2:
        query_interface(rag_pipeline, selected_model)
    
    # Footer
    st.markdown("---")
    st.markdown("*Powered by Streamlit Cloud | Legal Document Analyzer*")

if __name__ == "__main__":
    main()

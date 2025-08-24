"""
Enhanced Streamlit UI with full ML features for Legal Document Analyzer
"""
import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer - ML Enhanced",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API Configuration
API_BASE_URL = "http://localhost:8000"

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f4e79;
        margin: 0.5rem 0;
    }
    .confidence-high { color: #28a745; font-weight: bold; }
    .confidence-medium { color: #ffc107; font-weight: bold; }
    .confidence-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is running and get health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def upload_document(file):
    """Upload document to the API"""
    try:
        files = {"file": (file.name, file.getvalue(), "application/pdf")}
        response = requests.post(f"{API_BASE_URL}/upload", files=files, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Upload failed: {response.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def query_documents(question, max_results=5):
    """Query the documents"""
    try:
        payload = {
            "question": question,
            "max_results": max_results
        }
        response = requests.post(f"{API_BASE_URL}/query", json=payload, timeout=30)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, {"error": f"Query failed: {response.text}"}
    except Exception as e:
        return False, {"error": str(e)}

def get_statistics():
    """Get system statistics"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def get_documents():
    """Get list of processed documents"""
    try:
        response = requests.get(f"{API_BASE_URL}/documents", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def clear_all_documents():
    """Clear all documents from the system"""
    try:
        response = requests.delete(f"{API_BASE_URL}/documents", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def format_confidence(confidence):
    """Format confidence score with color coding"""
    if confidence >= 0.7:
        return f'<span class="confidence-high">{confidence:.1%}</span>'
    elif confidence >= 0.4:
        return f'<span class="confidence-medium">{confidence:.1%}</span>'
    else:
        return f'<span class="confidence-low">{confidence:.1%}</span>'

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">⚖️ Legal Document Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<div class="feature-box"><h3>🤖 ML-Powered Document Analysis with RAG Pipeline</h3></div>', unsafe_allow_html=True)
    
    # Check API health
    api_healthy, health_data = check_api_health()
    
    if not api_healthy:
        st.error("🚨 API is not responding. Please make sure the backend is running on http://localhost:8000")
        st.info("Run: `python src/api/main_ml.py` to start the backend")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 System Status")
        
        if health_data:
            ml_status = health_data.get("ml_status", {})
            st.success("✅ API Connected")
            
            # ML Status
            st.subheader("🧠 ML Components")
            st.write(f"**Vector DB:** {'✅ Active' if ml_status.get('vector_db_available') else '❌ Inactive'}")
            st.write(f"**LLM Model:** {ml_status.get('llm_model', 'Unknown')}")
            st.write(f"**Embedding Model:** {ml_status.get('embedding_model', 'Unknown')}")
            
            # Quick stats
            st.subheader("📊 Quick Stats")
            st.metric("Documents", ml_status.get('total_documents', 0))
            avg_conf = ml_status.get('average_confidence', 0)
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        
        # Clear documents button
        if st.button("🗑️ Clear All Documents", type="secondary"):
            success, result = clear_all_documents()
            if success:
                st.success("All documents cleared!")
                st.rerun()
            else:
                st.error("Failed to clear documents")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "❓ Query Documents", "📊 Analytics Dashboard", "⚙️ System Info"])
    
    # Tab 1: Document Upload
    with tab1:
        st.header("📄 Upload Legal Documents")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload legal documents like contracts, NDAs, service agreements, etc."
            )
            
            if uploaded_file is not None:
                st.info(f"File: {uploaded_file.name} ({uploaded_file.size:,} bytes)")
                
                if st.button("🚀 Process Document", type="primary"):
                    with st.spinner("Processing document... This may take a moment."):
                        success, result = upload_document(uploaded_file)
                        
                        if success:
                            st.success(f"✅ Document processed successfully!")
                            st.info(f"📚 Created {result['chunks_processed']} text chunks for analysis")
                            
                            # Show processing details
                            with st.expander("Processing Details"):
                                st.json(result)
                        else:
                            st.error(f"❌ Upload failed: {result.get('error', 'Unknown error')}")
        
        with col2:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - **Supported formats:** PDF only
            - **Best results:** Clear, text-based PDFs
            - **Processing:** Documents are split into chunks for optimal retrieval
            - **ML Features:** Uses sentence transformers for embeddings
            """)
    
    # Tab 2: Query Documents
    with tab2:
        st.header("❓ Query Your Documents")
        
        # Get documents list for context
        docs_success, docs_data = get_documents()
        
        if docs_success and docs_data.get('total_documents', 0) > 0:
            st.success(f"📚 {docs_data['total_documents']} documents ready for analysis")
            
            # Example questions
            with st.expander("📝 Example Questions"):
                example_questions = [
                    "What are the key terms and conditions in this contract?",
                    "What are the termination clauses mentioned?",
                    "Are there any liability limitations specified?",
                    "What are the payment terms and conditions?",
                    "What intellectual property rights are mentioned?",
                    "Are there any confidentiality clauses?",
                    "What are the dispute resolution mechanisms?",
                    "What governing law applies to this agreement?"
                ]
                
                for i, question in enumerate(example_questions, 1):
                    if st.button(f"{i}. {question}", key=f"example_{i}"):
                        st.session_state.query_text = question
            
            # Query input
            query_text = st.text_area(
                "Enter your legal question:",
                value=st.session_state.get('query_text', ''),
                height=100,
                placeholder="E.g., What are the termination conditions mentioned in the contract?"
            )
            
            col1, col2 = st.columns([3, 1])
            with col1:
                max_results = st.slider("Maximum results", 1, 10, 5)
            with col2:
                query_button = st.button("🔍 Analyze", type="primary", disabled=not query_text.strip())
            
            if query_button and query_text.strip():
                with st.spinner("🧠 Analyzing documents..."):
                    success, result = query_documents(query_text, max_results)
                    
                    if success:
                        st.subheader("🎯 Analysis Results")
                        
                        # Main answer
                        confidence = result.get('confidence', 0)
                        st.markdown(f"**Confidence:** {format_confidence(confidence)}", unsafe_allow_html=True)
                        
                        # Answer in a nice box
                        st.markdown("### 💡 Answer")
                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 1rem; border-radius: 8px; border-left: 4px solid #1f4e79;">
                        {result['answer']}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Model info
                        model_info = result.get('model_info', {})
                        processing_time = result.get('processing_time', 0)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        with col2:
                            st.metric("Model", model_info.get('model', 'Unknown'))
                        with col3:
                            st.metric("Tokens Used", model_info.get('tokens_used', 0))
                        
                        # Sources
                        sources = result.get('sources', [])
                        if sources:
                            st.subheader("📚 Source Documents")
                            for i, source in enumerate(sources, 1):
                                with st.expander(f"Source {i} (Score: {source.get('score', 0):.3f})"):
                                    st.write(source.get('content', 'No content available'))
                                    
                                    metadata = source.get('metadata', {})
                                    if metadata:
                                        st.json(metadata)
                    else:
                        st.error(f"❌ Query failed: {result.get('error', 'Unknown error')}")
        else:
            st.warning("📭 No documents uploaded yet. Please upload documents first.")
            st.info("👆 Go to the 'Document Upload' tab to add documents for analysis.")
    
    # Tab 3: Analytics Dashboard
    with tab3:
        st.header("📊 Analytics Dashboard")
        
        stats_success, stats_data = get_statistics()
        
        if stats_success and stats_data:
            # Key metrics
            st.subheader("🎯 Key Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_docs = stats_data.get('api', {}).get('documents_uploaded', 0)
                st.metric("Documents Uploaded", total_docs)
            
            with col2:
                total_chunks = stats_data.get('database', {}).get('total_documents', 0)
                st.metric("Text Chunks", total_chunks)
            
            with col3:
                total_queries = stats_data.get('api', {}).get('total_queries', 0)
                st.metric("Total Queries", total_queries)
            
            with col4:
                avg_confidence = stats_data.get('database', {}).get('average_confidence', 0)
                st.metric("Avg Confidence", f"{avg_confidence:.1%}")
            
            # Recent activity
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Recent Uploads")
                recent_uploads = stats_data.get('api', {}).get('recent_uploads', [])
                if recent_uploads:
                    df_uploads = pd.DataFrame(recent_uploads)
                    st.dataframe(df_uploads, use_container_width=True)
                else:
                    st.info("No recent uploads")
            
            with col2:
                st.subheader("🔍 Recent Queries")
                recent_queries = stats_data.get('recent_queries', [])
                if recent_queries:
                    df_queries = pd.DataFrame(recent_queries)
                    # Create confidence chart
                    if 'confidence' in df_queries.columns:
                        fig = px.bar(
                            df_queries.tail(10), 
                            y='confidence',
                            title="Query Confidence Scores",
                            color='confidence',
                            color_continuous_scale='viridis'
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.dataframe(df_queries, use_container_width=True)
                else:
                    st.info("No recent queries")
            
            # Configuration
            st.subheader("⚙️ System Configuration")
            config = stats_data.get('config', {})
            
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Chunk Size:** {config.get('max_chunk_size', 'Unknown')}")
                st.write(f"**Chunk Overlap:** {config.get('chunk_overlap', 'Unknown')}")
            with col2:
                st.write(f"**LLM Model:** {config.get('default_llm', 'Unknown')}")
                st.write(f"**Embedding Model:** {config.get('embedding_model', 'Unknown')}")
        
        else:
            st.error("Failed to load analytics data")
    
    # Tab 4: System Info
    with tab4:
        st.header("⚙️ System Information")
        
        # API Health
        if health_data:
            st.subheader("🏥 Health Status")
            st.json(health_data)
        
        # Full statistics
        if stats_success:
            st.subheader("📊 Detailed Statistics")
            st.json(stats_data)
        
        # Documents list
        docs_success, docs_data = get_documents()
        if docs_success:
            st.subheader("📄 Document Registry")
            st.json(docs_data)
        
        # API Endpoints
        st.subheader("🔗 API Endpoints")
        endpoints = [
            {"Method": "GET", "Endpoint": "/", "Description": "API root"},
            {"Method": "POST", "Endpoint": "/upload", "Description": "Upload document"},
            {"Method": "POST", "Endpoint": "/query", "Description": "Query documents"},
            {"Method": "GET", "Endpoint": "/health", "Description": "Health check"},
            {"Method": "GET", "Endpoint": "/stats", "Description": "System statistics"},
            {"Method": "GET", "Endpoint": "/documents", "Description": "List documents"},
            {"Method": "DELETE", "Endpoint": "/documents", "Description": "Clear all documents"}
        ]
        st.dataframe(pd.DataFrame(endpoints), use_container_width=True)

if __name__ == "__main__":
    main()

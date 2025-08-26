"""
Simplified configuration for Streamlit Cloud deployment.
"""
import os
from pathlib import Path

class CloudSettings:
    """Settings optimized for Streamlit Cloud."""
    
    # Core paths
    base_dir = Path(".")
    data_dir = base_dir / "data"
    upload_path = data_dir / "uploads"
    vector_db_path = data_dir / "vector_db"
    
    # Model settings
    default_llm = "huggingface"
    hf_model_name = "google/flan-t5-large"
    hf_use_inference_api = False
    
    # Processing settings
    chunk_size = 500
    chunk_overlap = 50
    max_content_length = 4000
    
    # LLM settings
    llm_temperature = 0.7
    max_tokens = 512
    confidence_threshold = 0.3
    
    # Vector DB settings
    embedding_model = "all-MiniLM-L6-v2"
    vector_store_type = "chromadb"
    
    # Cloud optimizations
    enable_monitoring = False
    log_level = "INFO"
    max_upload_size = 200  # MB
    
    def __init__(self):
        """Create necessary directories."""
        self.data_dir.mkdir(exist_ok=True)
        self.upload_path.mkdir(exist_ok=True)
        self.vector_db_path.mkdir(exist_ok=True)

# Global settings instance
settings = CloudSettings()

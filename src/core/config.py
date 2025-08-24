"""
Configuration management for the Legal Document Analyzer.
"""
import os
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings."""
    
    # API Keys
    openai_api_key: Optional[str] = Field(None, env="OPENAI_API_KEY")
    huggingface_token: Optional[str] = Field(None, env="HUGGINGFACE_TOKEN")
    wandb_api_key: Optional[str] = Field(None, env="WANDB_API_KEY")
    
    # Vector Database Configuration
    vector_db_path: str = Field("./data/vector_db", env="VECTOR_DB_PATH")
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")
    
    # LLM Configuration
    default_llm: str = Field("huggingface", env="DEFAULT_LLM")
    llm_temperature: float = Field(0.3, env="LLM_TEMPERATURE")
    max_tokens: int = Field(2048, env="MAX_TOKENS")
    
    # Hugging Face Configuration
    hf_model_name: str = Field("google/flan-t5-base", env="HF_MODEL_NAME")
    hf_use_inference_api: bool = Field(False, env="HF_USE_INFERENCE_API")
    
    # API Configuration
    api_host: str = Field("localhost", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    cors_origins: List[str] = Field(
        ["http://localhost:8501", "http://localhost:3000"], 
        env="CORS_ORIGINS"
    )
    
    # Streamlit Configuration
    streamlit_port: int = Field(8501, env="STREAMLIT_PORT")
    
    # Logging
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("./logs/app.log", env="LOG_FILE")
    
    # Monitoring
    enable_monitoring: bool = Field(True, env="ENABLE_MONITORING")
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")
    
    # Data paths
    upload_path: str = "./data/uploads"
    cuad_data_path: str = "./data/cuad"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()

# Ensure directories exist
os.makedirs(settings.vector_db_path, exist_ok=True)
os.makedirs(settings.upload_path, exist_ok=True)
os.makedirs(settings.cuad_data_path, exist_ok=True)
os.makedirs(os.path.dirname(settings.log_file), exist_ok=True)

"""
Production configuration for the Legal Document Analyzer.
"""
import os
from src.core.config import Settings


class ProductionSettings(Settings):
    """Production-specific settings."""
    
    # Security
    cors_origins: list = ["https://yourdomain.com"]
    
    # Performance
    chunk_size: int = 500  # Smaller chunks for better precision
    chunk_overlap: int = 50
    
    # LLM
    default_llm: str = "openai"  # Use OpenAI for production
    llm_temperature: float = 0.05  # Lower temperature for more consistent results
    
    # Monitoring
    enable_monitoring: bool = True
    confidence_threshold: float = 0.8  # Higher threshold for production
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env.production"


class DevelopmentSettings(Settings):
    """Development-specific settings."""
    
    # Security (relaxed for development)
    cors_origins: list = ["*"]
    
    # LLM (can use local models for development)
    default_llm: str = "llama"
    llm_temperature: float = 0.2
    
    # Monitoring
    enable_monitoring: bool = False  # Disable for development
    confidence_threshold: float = 0.5
    
    # Logging
    log_level: str = "DEBUG"
    
    class Config:
        env_file = ".env.development"


def get_settings():
    """Get settings based on environment."""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionSettings()
    else:
        return DevelopmentSettings()

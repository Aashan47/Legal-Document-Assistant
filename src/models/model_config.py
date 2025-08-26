"""
Model configuration and selection for Legal Document Analyzer.
"""
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_id: str
    description: str
    strengths: str
    best_for: str
    speed: str  # Fast, Medium, Slow
    quality: str  # Good, Very Good, Excellent
    size: str  # Small, Medium, Large
    use_api: bool = False

# Available models configuration
AVAILABLE_MODELS = {
    "flan-t5-large": ModelConfig(
        name="FLAN-T5 Large",
        model_id="google/flan-t5-large",
        description="Google's instruction-tuned model, excellent for Q&A and analysis",
        strengths="Excellent instruction following, detailed analysis, good reasoning",
        best_for="Complex legal analysis, detailed contract review, multi-step reasoning",
        speed="Medium",
        quality="Very Good",
        size="Large (780M)",
        use_api=False
    ),
    "flan-t5-base": ModelConfig(
        name="FLAN-T5 Base",
        model_id="google/flan-t5-base",
        description="Balanced model for quick analysis and general queries",
        strengths="Fast responses, good general understanding, reliable performance",
        best_for="Quick document summaries, simple Q&A, general legal queries",
        speed="Fast",
        quality="Good",
        size="Medium (250M)",
        use_api=False
    ),
    "flan-t5-small": ModelConfig(
        name="FLAN-T5 Small",
        model_id="google/flan-t5-small",
        description="Compact and fast model for quick legal document queries",
        strengths="Very fast responses, lightweight, good for basic analysis",
        best_for="Quick document summaries, simple questions, basic legal queries",
        speed="Very Fast",
        quality="Good",
        size="Small (80M)",
        use_api=False
    )
}

def get_model_by_id(model_id: str) -> ModelConfig:
    """Get model config by model ID."""
    for config in AVAILABLE_MODELS.values():
        if config.model_id == model_id:
            return config
    # Return default if not found
    return AVAILABLE_MODELS["flan-t5-large"]

def get_models_by_speed(speed: str) -> List[ModelConfig]:
    """Get models filtered by speed."""
    return [config for config in AVAILABLE_MODELS.values() if config.speed == speed]

def get_models_by_quality(quality: str) -> List[ModelConfig]:
    """Get models filtered by quality."""
    return [config for config in AVAILABLE_MODELS.values() if config.quality == quality]

def get_recommended_model(use_case: str) -> ModelConfig:
    """Get recommended model for specific use case."""
    use_case_lower = use_case.lower()
    
    if "quick" in use_case_lower or "fast" in use_case_lower or "summary" in use_case_lower:
        return AVAILABLE_MODELS["flan-t5-small"]
    elif "complex" in use_case_lower or "detailed" in use_case_lower or "compliance" in use_case_lower:
        return AVAILABLE_MODELS["flan-t5-large"]
    else:
        return AVAILABLE_MODELS["flan-t5-base"]  # Default balanced choice

"""
Model Manager for pre-loading and caching all available models.
"""
import os
import torch
from typing import Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
import pickle
from pathlib import Path

from src.models.model_config import AVAILABLE_MODELS, ModelConfig
from src.utils.logging import app_logger
from src.core.config import settings


class ModelManager:
    """Manages pre-loading, caching, and serving of all available models."""
    
    def __init__(self, cache_dir: str = "./data/model_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory model cache
        self._model_cache: Dict[str, Dict] = {}
        self._current_model: Optional[str] = None
        self._current_tokenizer = None
        self._current_model_obj = None
        self._current_model_type = None
        
        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        app_logger.info(f"ModelManager initialized with device: {self.device}")
    
    def get_model_cache_path(self, model_id: str) -> tuple:
        """Get cache file paths for a model."""
        safe_name = model_id.replace("/", "_").replace("-", "_")
        model_path = self.cache_dir / f"{safe_name}_model.pkl"
        tokenizer_path = self.cache_dir / f"{safe_name}_tokenizer.pkl"
        config_path = self.cache_dir / f"{safe_name}_config.pkl"
        return model_path, tokenizer_path, config_path
    
    def determine_model_type(self, model_id: str) -> str:
        """Determine if model is seq2seq or causal LM."""
        try:
            config = AutoConfig.from_pretrained(model_id)
            
            # Check for seq2seq models
            seq2seq_architectures = [
                "T5Config", "BartConfig", "PegasusConfig", "MarianConfig", 
                "MT5Config", "FlanT5Config", "UL2Config"
            ]
            
            # Check for causal LM models  
            causal_lm_architectures = [
                "GPT2Config", "GPTNeoConfig", "GPTJConfig", "LlamaConfig",
                "BloomConfig", "OPTConfig"
            ]
            
            config_class_name = config.__class__.__name__
            
            if any(arch in config_class_name for arch in seq2seq_architectures):
                return "seq2seq"
            elif any(arch in config_class_name for arch in causal_lm_architectures):
                return "causal_lm"
            elif hasattr(config, 'is_encoder_decoder') and config.is_encoder_decoder:
                return "seq2seq"
            else:
                # Default to causal LM for unknown types
                app_logger.warning(f"Unknown model type for {model_id}, defaulting to causal_lm")
                return "causal_lm"
                
        except Exception as e:
            app_logger.error(f"Error determining model type for {model_id}: {e}")
            return "causal_lm"  # Default fallback
    
    def is_model_cached(self, model_id: str) -> bool:
        """Check if model is already cached on disk."""
        model_path, tokenizer_path, config_path = self.get_model_cache_path(model_id)
        return model_path.exists() and tokenizer_path.exists() and config_path.exists()
    
    def cache_model_to_disk(self, model_id: str) -> bool:
        """Download and cache a model to disk."""
        try:
            app_logger.info(f"Downloading and caching model: {model_id}")
            
            # Determine model type
            model_type = self.determine_model_type(model_id)
            app_logger.info(f"Model {model_id} identified as: {model_type}")
            
            # Load tokenizer and model based on type
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            
            # Add pad token if it doesn't exist (for causal LM models)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
            else:  # causal_lm
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
            
            # Get cache paths
            model_path, tokenizer_path, config_path = self.get_model_cache_path(model_id)
            
            # Save to disk
            with open(model_path, 'wb') as f:
                pickle.dump(model.state_dict(), f)
            
            with open(tokenizer_path, 'wb') as f:
                pickle.dump(tokenizer, f)
                
            with open(config_path, 'wb') as f:
                pickle.dump({"model_type": model_type, "model_id": model_id}, f)
            
            app_logger.info(f"Successfully cached model {model_id} to disk")
            return True
            
        except Exception as e:
            app_logger.error(f"Error caching model {model_id}: {str(e)}")
            return False
    
    def load_model_from_cache(self, model_id: str) -> tuple:
        """Load a model from disk cache."""
        try:
            model_path, tokenizer_path, config_path = self.get_model_cache_path(model_id)
            
            if not (model_path.exists() and tokenizer_path.exists() and config_path.exists()):
                raise FileNotFoundError(f"Cached model not found for {model_id}")
            
            app_logger.info(f"Loading cached model: {model_id}")
            
            # Load config to determine model type
            with open(config_path, 'rb') as f:
                config_data = pickle.load(f)
            
            model_type = config_data.get("model_type", "causal_lm")
            
            # Load tokenizer
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            
            # Create model instance based on type
            if model_type == "seq2seq":
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
            else:  # causal_lm
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
                    device_map="auto" if self.device.type == "cuda" else None
                )
            
            # Load cached state dict
            with open(model_path, 'rb') as f:
                state_dict = pickle.load(f)
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            app_logger.info(f"Successfully loaded cached model: {model_id}")
            return tokenizer, model, model_type
            
        except Exception as e:
            app_logger.error(f"Error loading cached model {model_id}: {str(e)}")
            return None, None, None
    
    def preload_all_models(self) -> bool:
        """Pre-load and cache all available models."""
        app_logger.info("Starting to pre-load all available models...")
        
        success_count = 0
        total_models = len(AVAILABLE_MODELS)
        
        for model_key, config in AVAILABLE_MODELS.items():
            model_id = config.model_id
            app_logger.info(f"Processing model {success_count + 1}/{total_models}: {model_id}")
            
            try:
                # Check if already cached
                if self.is_model_cached(model_id):
                    app_logger.info(f"Model {model_id} already cached, skipping download")
                    success_count += 1
                    continue
                
                # Cache the model
                if self.cache_model_to_disk(model_id):
                    success_count += 1
                    app_logger.info(f"Successfully processed {model_id}")
                else:
                    app_logger.warning(f"Failed to cache {model_id}")
                    
            except Exception as e:
                app_logger.error(f"Error processing model {model_id}: {str(e)}")
        
        app_logger.info(f"Pre-loading complete: {success_count}/{total_models} models cached")
        return success_count == total_models
    
    def load_model_for_inference(self, model_id: str) -> bool:
        """Load a specific model for inference (switching models)."""
        try:
            # If it's already the current model, no need to reload
            if self._current_model == model_id and self._current_model_obj is not None:
                app_logger.info(f"Model {model_id} already loaded")
                return True
            
            # Clear current model from memory
            if self._current_model_obj is not None:
                del self._current_model_obj
                del self._current_tokenizer
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # Load from cache if available
            if self.is_model_cached(model_id):
                tokenizer, model, model_type = self.load_model_from_cache(model_id)
                if tokenizer is not None and model is not None:
                    self._current_tokenizer = tokenizer
                    self._current_model_obj = model
                    self._current_model = model_id
                    self._current_model_type = model_type
                    app_logger.info(f"Switched to cached model: {model_id}")
                    return True
            
            # If not cached, download and cache first
            app_logger.info(f"Model {model_id} not cached, downloading...")
            if self.cache_model_to_disk(model_id):
                return self.load_model_for_inference(model_id)  # Retry after caching
            
            return False
            
        except Exception as e:
            app_logger.error(f"Error loading model for inference {model_id}: {str(e)}")
            return False
    
    def generate_response(self, question: str, context: list, model_id: str = None) -> dict:
        """Generate response using the specified or current model."""
        try:
            # Load model if specified and different from current
            if model_id and model_id != self._current_model:
                if not self.load_model_for_inference(model_id):
                    raise Exception(f"Failed to load model {model_id}")
            
            # Ensure we have a model loaded
            if self._current_model_obj is None or self._current_tokenizer is None:
                # Default to first available model
                default_model = list(AVAILABLE_MODELS.values())[0].model_id
                if not self.load_model_for_inference(default_model):
                    raise Exception("Failed to load any model")
            
            # Prepare prompt based on model type
            context_text = "\n\n".join(context[:3])  # Limit context to avoid token limits
            
            if self._current_model_type == "seq2seq":
                # Seq2seq models (T5, BART, etc.)
                prompt = f"""Based on the following legal document excerpts, provide a detailed and comprehensive answer to the question.

Legal Documents:
{context_text}

Question: {question}

Please provide a thorough analysis based on the document content, including specific references to relevant clauses or sections when applicable."""

                # Tokenize and generate
                inputs = self._current_tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self._current_model_obj.generate(
                        **inputs,
                        max_new_tokens=512,
                        min_length=50,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=self._current_tokenizer.eos_token_id
                    )
                
                # Decode response
                response = self._current_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            else:  # causal_lm (GPT-2, DialoGPT, etc.)
                # Causal LM models expect different prompting
                prompt = f"""Legal Document Analysis:

Documents: {context_text}

Question: {question}

Answer:"""

                # Tokenize
                inputs = self._current_tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=1024,
                    truncation=True,
                    padding=True
                ).to(self.device)
                
                # Generate with attention mask
                with torch.no_grad():
                    outputs = self._current_model_obj.generate(
                        input_ids=inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        pad_token_id=self._current_tokenizer.eos_token_id,
                        eos_token_id=self._current_tokenizer.eos_token_id
                    )
                
                # Decode only the new tokens (remove input)
                input_length = inputs['input_ids'].shape[1]
                new_tokens = outputs[0][input_length:]
                response = self._current_tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            # Clean up response
            if "Answer:" in response:
                answer = response.split("Answer:")[-1].strip()
            else:
                answer = response.strip()
            
            # Ensure minimum answer length
            if len(answer) < 20:
                answer = "Based on the provided legal documents, I can analyze the relevant provisions. However, I need more specific information to provide a detailed response. Please consider uploading more relevant documents or refining your question."
            
            # Calculate confidence based on response length and content
            confidence = min(0.9, max(0.3, len(answer) / 200))
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model": self._current_model,
                "tokens_used": len(outputs[0]) if 'outputs' in locals() else 0
            }
            
        except Exception as e:
            app_logger.error(f"Error generating response: {str(e)}")
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again with a different model or rephrase your question.",
                "confidence": 0.0,
                "model": self._current_model or "unknown",
                "tokens_used": 0
            }
    
    def get_cache_info(self) -> dict:
        """Get information about cached models."""
        cached_models = []
        total_size = 0
        
        for model_key, config in AVAILABLE_MODELS.items():
            model_id = config.model_id
            if self.is_model_cached(model_id):
                model_path, tokenizer_path, config_path = self.get_model_cache_path(model_id)
                size = model_path.stat().st_size + tokenizer_path.stat().st_size + config_path.stat().st_size
                total_size += size
                cached_models.append({
                    "model_id": model_id,
                    "model_name": config.name,
                    "size_mb": size / (1024 * 1024)
                })
        
        return {
            "cached_models": cached_models,
            "total_cached": len(cached_models),
            "total_available": len(AVAILABLE_MODELS),
            "total_size_gb": total_size / (1024 * 1024 * 1024),
            "current_model": self._current_model
        }


# Global model manager instance
_model_manager = None

def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager

def initialize_model_cache():
    """Initialize and pre-load all models."""
    manager = get_model_manager()
    return manager.preload_all_models()

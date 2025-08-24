"""
LLM integrations for the Legal Document Analyzer.
Supports OpenAI GPT-4, Llama models, and Hugging Face models.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import openai
from src.core.config import settings
from src.utils.logging import app_logger


class BaseLLM(ABC):
    """Abstract base class for LLM implementations."""
    
    @abstractmethod
    def generate_response(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        """Generate a response given a prompt and context."""
        pass
    
    @abstractmethod
    def calculate_confidence(self, response: str, context: List[str]) -> float:
        """Calculate confidence score for the response."""
        pass


class OpenAILLM(BaseLLM):
    """OpenAI GPT-4 implementation."""
    
    def __init__(self):
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not provided")
        
        openai.api_key = settings.openai_api_key
        self.model = "gpt-4"
        self.temperature = settings.llm_temperature
        self.max_tokens = settings.max_tokens
    
    def generate_response(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        """Generate response using OpenAI GPT-4."""
        try:
            # Prepare the conversation
            system_message = self._create_system_message()
            user_message = self._create_user_message(prompt, context)
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_message}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            answer = response.choices[0].message.content
            confidence = self.calculate_confidence(answer, context)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model": self.model,
                "tokens_used": response.usage.total_tokens
            }
            
        except Exception as e:
            app_logger.error(f"Error generating OpenAI response: {str(e)}")
            raise
    
    def calculate_confidence(self, response: str, context: List[str]) -> float:
        """Calculate confidence score based on context relevance."""
        try:
            # Simple confidence calculation based on context overlap
            if not context or not response:
                return 0.0
            
            response_words = set(response.lower().split())
            context_words = set()
            for ctx in context:
                context_words.update(ctx.lower().split())
            
            if not context_words:
                return 0.5  # Medium confidence if no context
            
            overlap = len(response_words.intersection(context_words))
            confidence = min(overlap / len(response_words), 1.0)
            
            # Boost confidence for longer, more detailed responses
            length_factor = min(len(response) / 500, 1.0)
            confidence = (confidence + length_factor) / 2
            
            return max(0.1, confidence)  # Minimum confidence of 0.1
            
        except Exception as e:
            app_logger.error(f"Error calculating confidence: {str(e)}")
            return 0.5
    
    def _create_system_message(self) -> str:
        """Create system message for the LLM."""
        return """You are an expert legal document analyzer. Your task is to answer questions about legal documents based on the provided context.

Guidelines:
1. Only answer based on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Cite specific sections or clauses when possible
4. Be precise and professional in your language
5. Highlight potential legal risks or important clauses
6. If you're unsure, express that uncertainty

Always structure your response clearly and provide reasoning for your analysis."""
    
    def _create_user_message(self, prompt: str, context: List[str]) -> str:
        """Create user message with prompt and context."""
        context_text = "\n\n".join([f"Document {i+1}:\n{ctx}" for i, ctx in enumerate(context)])
        
        return f"""Context from legal documents:
{context_text}

Question: {prompt}

Please analyze the provided context and answer the question. If the context doesn't contain sufficient information to answer the question, please state that clearly."""


class HuggingFaceLLM(BaseLLM):
    """Hugging Face model implementation for open-source LLMs."""
    
    def __init__(self, model_name: str = None, use_inference_api: bool = True):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, T5Tokenizer, T5ForConditionalGeneration
            import torch
            
            self.model_name = model_name or settings.hf_model_name or "google/flan-t5-base"
            self.use_inference_api = use_inference_api
            self.temperature = settings.llm_temperature
            self.max_tokens = settings.max_tokens
            
            if self.use_inference_api:
                # Use Hugging Face Inference API
                if "t5" in self.model_name.lower():
                    self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                else:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = None  # Will use API calls
                app_logger.info(f"Using HF Inference API for model: {self.model_name}")
            else:
                # Load model locally
                app_logger.info(f"Loading local HF model: {self.model_name}")
                
                if "t5" in self.model_name.lower():
                    # Use T5 tokenizer and model for T5-based models
                    self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
                    self.model = T5ForConditionalGeneration.from_pretrained(
                        self.model_name,
                        device_map="auto" if torch.cuda.is_available() else None,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        low_cpu_mem_usage=True
                    )
                else:
                    # Use standard tokenizer for other models
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    
                    # Add pad token if not present
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token
                    
                    # Load model with optimizations for large models
                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        device_map="auto" if torch.cuda.is_available() else None,
                        low_cpu_mem_usage=True,
                        trust_remote_code=True,  # Required for some models
                        load_in_8bit=True if torch.cuda.is_available() else False,  # Memory optimization
                        use_cache=True
                    )
                app_logger.info(f"Loaded local HF model: {self.model_name}")
            
        except ImportError:
            raise ImportError("transformers not installed. Install with: pip install transformers torch")
        except Exception as e:
            app_logger.error(f"Error loading HF model: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        """Generate response using Hugging Face model."""
        try:
            full_prompt = self._create_prompt(prompt, context)
            
            if self.use_inference_api:
                # Use Hugging Face Inference API
                response = self._generate_with_api(full_prompt)
            else:
                # Use local model
                response = self._generate_with_local_model(full_prompt)
            
            confidence = self.calculate_confidence(response, context)
            
            return {
                "answer": response,
                "confidence": confidence,
                "model": self.model_name,
                "tokens_used": len(full_prompt.split()) + len(response.split())
            }
            
        except Exception as e:
            app_logger.error(f"Error generating HF response: {str(e)}")
            raise
    
    def _generate_with_api(self, prompt: str) -> str:
        """Generate response using Hugging Face Inference API."""
        try:
            import requests
            
            api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
            headers = {"Authorization": f"Bearer {settings.huggingface_token}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.temperature,
                    "max_new_tokens": min(self.max_tokens, 1000),  # API limits
                    "min_length": 50,  # Ensure minimum response length
                    "top_p": 0.9,
                    "top_k": 50,
                    "repetition_penalty": 1.2,
                    "length_penalty": 1.0,
                    "return_full_text": False
                }
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                generated_text = result.get("generated_text", "")
            else:
                generated_text = "Unable to generate response"
            
            return generated_text.strip()
            
        except Exception as e:
            app_logger.error(f"Error with HF Inference API: {str(e)}")
            return "Error generating response with Hugging Face API"
    
    def _generate_with_local_model(self, prompt: str) -> str:
        """Generate response using local Hugging Face model."""
        try:
            import torch
            
            # Check if this is a T5 model
            if "t5" in self.model_name.lower():
                # T5 models use encoder-decoder architecture
                # Format the input for T5
                input_text = f"Answer the following question based on the context: {prompt}"
                input_ids = self.tokenizer(
                    input_text, 
                    return_tensors="pt", 
                    max_length=512, 
                    truncation=True
                ).input_ids
                
                # Move to device if available
                if torch.cuda.is_available() and hasattr(self.model, 'device'):
                    input_ids = input_ids.to(self.model.device)
                
                # Generate response with T5
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_new_tokens=min(self.max_tokens, 2048),
                        min_length=50,  # Ensure minimum response length
                        temperature=self.temperature,
                        do_sample=True,
                        top_p=0.9,  # Use nucleus sampling for better quality
                        top_k=50,   # Limit vocabulary for coherence
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.2,  # Reduce repetition
                        length_penalty=1.0,      # Encourage longer responses
                        no_repeat_ngram_size=3   # Avoid repetitive n-grams
                    )
                
                # Decode the response
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
            # For OpenAI GPT-OSS models that support chat templates
            elif "openai/gpt-oss" in self.model_name.lower():
                # Create messages format for chat models
                messages = [
                    {"role": "system", "content": "You are a legal document analysis expert. Provide accurate, detailed analysis based on the given context."},
                    {"role": "user", "content": prompt}
                ]
                
                # Apply chat template
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device if hasattr(self.model, 'device') else 'cpu')
                
                # Generate response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=min(self.max_tokens, 512),
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.1
                    )
                
                # Decode only the new tokens (response)
                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[-1]:], 
                    skip_special_tokens=True
                )
                
            else:
                # Standard approach for other models
                inputs = self.tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=2048)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=min(self.max_tokens, 512),
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove the input prompt from the response
                if response.startswith(prompt):
                    response = response[len(prompt):].strip()
            
            return response.strip()
            
        except Exception as e:
            app_logger.error(f"Error with local HF model: {str(e)}")
            return "Error generating response with local model"
    
    def calculate_confidence(self, response: str, context: List[str]) -> float:
        """Calculate confidence score for HF response."""
        try:
            if not context or not response:
                return 0.0
            
            response_words = set(response.lower().split())
            context_words = set()
            for ctx in context:
                context_words.update(ctx.lower().split())
            
            if not context_words:
                return 0.5
            
            overlap = len(response_words.intersection(context_words))
            confidence = min(overlap / len(response_words), 1.0) if response_words else 0.0
            
            # Boost confidence for coherent responses
            if len(response) > 50 and "error" not in response.lower():
                confidence = min(confidence + 0.1, 1.0)
            
            return max(0.1, confidence)
            
        except Exception as e:
            app_logger.error(f"Error calculating HF confidence: {str(e)}")
            return 0.5
    
    def _create_prompt(self, prompt: str, context: List[str]) -> str:
        """Create formatted prompt for Hugging Face models."""
        context_text = "\n\n".join(context[:3])  # Limit context for token limits
        
        # Create a legal-specific prompt template
        if "t5" in self.model_name.lower():
            # T5 models work better with detailed task-specific prompts
            return f"""You are an expert legal document analyzer. Provide a comprehensive analysis of the legal document based on the context provided.

Context from legal documents:
{context_text}

Question: {prompt}

Please provide a detailed analysis that includes:
1. Key terms and conditions mentioned
2. Important clauses and their implications
3. Specific legal provisions
4. Relevant sections or paragraphs
5. Any potential risks or considerations

Detailed answer:"""
        else:
            # Standard prompt for other models
            return f"""<s>[INST] You are an expert legal document analyzer. Based on the provided legal document context, answer the question accurately and professionally.

Context from legal documents:
{context_text}

Question: {prompt}

Please provide a detailed answer based only on the information in the context. If the context doesn't contain enough information, state that clearly. [/INST]

Answer: """


class LlamaLLM(BaseLLM):
    """Local Llama model implementation."""
    
    def __init__(self, model_path: str = None):
        try:
            from llama_cpp import Llama
            
            if not model_path:
                # Default model path - user should download the model
                model_path = "./models/llama-2-7b-chat.gguf"
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_threads=4,
                verbose=False
            )
            app_logger.info(f"Loaded Llama model from {model_path}")
            
        except ImportError:
            raise ImportError("llama-cpp-python not installed. Install with: pip install llama-cpp-python")
        except Exception as e:
            app_logger.error(f"Error loading Llama model: {str(e)}")
            raise
    
    def generate_response(self, prompt: str, context: List[str]) -> Dict[str, Any]:
        """Generate response using local Llama model."""
        try:
            full_prompt = self._create_prompt(prompt, context)
            
            response = self.llm(
                full_prompt,
                max_tokens=settings.max_tokens,
                temperature=settings.llm_temperature,
                echo=False
            )
            
            answer = response['choices'][0]['text'].strip()
            confidence = self.calculate_confidence(answer, context)
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model": "llama",
                "tokens_used": len(full_prompt.split()) + len(answer.split())
            }
            
        except Exception as e:
            app_logger.error(f"Error generating Llama response: {str(e)}")
            raise
    
    def calculate_confidence(self, response: str, context: List[str]) -> float:
        """Calculate confidence score for Llama response."""
        # Similar to OpenAI implementation
        try:
            if not context or not response:
                return 0.0
            
            response_words = set(response.lower().split())
            context_words = set()
            for ctx in context:
                context_words.update(ctx.lower().split())
            
            if not context_words:
                return 0.5
            
            overlap = len(response_words.intersection(context_words))
            confidence = min(overlap / len(response_words), 1.0) if response_words else 0.0
            
            return max(0.1, confidence)
            
        except Exception as e:
            app_logger.error(f"Error calculating Llama confidence: {str(e)}")
            return 0.5
    
    def _create_prompt(self, prompt: str, context: List[str]) -> str:
        """Create formatted prompt for Llama."""
        context_text = "\n\n".join(context)
        
        return f"""<s>[INST] <<SYS>>
You are an expert legal document analyzer. Answer questions about legal documents based only on the provided context.
<</SYS>>

Context: {context_text}

Question: {prompt} [/INST]"""


class LLMFactory:
    """Factory class for creating LLM instances."""
    
    @staticmethod
    def create_llm(llm_type: str = None) -> BaseLLM:
        """Create an LLM instance based on configuration."""
        if llm_type is None:
            llm_type = settings.default_llm
        
        if llm_type.lower() == "openai":
            return OpenAILLM()
        elif llm_type.lower() == "llama":
            return LlamaLLM()
        elif llm_type.lower() == "huggingface":
            model_name = getattr(settings, 'hf_model_name', 'microsoft/DialoGPT-medium')
            use_api = getattr(settings, 'hf_use_inference_api', True)
            return HuggingFaceLLM(model_name=model_name, use_inference_api=use_api)
        else:
            raise ValueError(f"Unsupported LLM type: {llm_type}")


# Global LLM instance - will be created when first accessed
_llm_instance = None

def get_llm():
    """Get the global LLM instance, creating it if necessary."""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMFactory.create_llm()
    return _llm_instance

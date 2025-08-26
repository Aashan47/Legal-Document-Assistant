"""
RAG (Retrieval-Augmented Generation) pipeline for legal document analysis.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib
import numpy as np

# ML imports
try:
    from sentence_transformers import SentenceTransformer
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

from src.core.vector_db import VectorDatabase
from src.models.llm import get_llm
from src.data.processor import DocumentProcessor
from src.utils.logging import app_logger
from src.core.config import settings


class RAGPipeline:
    """Main RAG pipeline for legal document analysis."""
    
    def __init__(self):
        self.vector_db = VectorDatabase()
        self.llm = get_llm()
        self.document_processor = DocumentProcessor()
        self.query_history = []
        
        # Initialize ML components if available
        self.embedding_model = None
        if ML_AVAILABLE:
            try:
                model_name = getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')
                self.embedding_model = SentenceTransformer(model_name)
                app_logger.info(f"Initialized embedding model: {model_name}")
            except Exception as e:
                app_logger.warning(f"Failed to initialize embedding model: {e}")
                self.embedding_model = None
    
    def add_documents(self, file_paths: List[str]) -> Dict[str, Any]:
        """Add documents to the knowledge base."""
        try:
            results = {
                "success": [],
                "failed": [],
                "total_chunks": 0
            }
            
            for file_path in file_paths:
                try:
                    # Validate file
                    if not self.document_processor.validate_file(file_path):
                        results["failed"].append({
                            "file": file_path,
                            "error": "Invalid PDF file"
                        })
                        continue
                    
                    # Process document
                    documents = self.document_processor.process_uploaded_file(file_path)
                    
                    # Add to vector database
                    self.vector_db.add_documents(documents)
                    
                    results["success"].append({
                        "file": file_path,
                        "chunks": len(documents)
                    })
                    results["total_chunks"] += len(documents)
                    
                except Exception as e:
                    results["failed"].append({
                        "file": file_path,
                        "error": str(e)
                    })
                    app_logger.error(f"Failed to process {file_path}: {str(e)}")
            
            app_logger.info(f"Added documents: {len(results['success'])} successful, {len(results['failed'])} failed")
            return results
            
        except Exception as e:
            app_logger.error(f"Error in add_documents: {str(e)}")
            raise
    
    def query(self, question: str, top_k: int = 5, model_name: str = None) -> Dict[str, Any]:
        """Query the knowledge base and generate an answer."""
        try:
            app_logger.info(f"Processing query: {question[:100]}...")
            
            # Import here to avoid circular imports
            from src.models.llm import get_llm
            
            # Get LLM instance (potentially different model)
            if model_name:
                app_logger.info(f"Using model: {model_name}")
                llm = get_llm(model_name=model_name)
            else:
                llm = self.llm
            
            # Retrieve relevant documents
            search_results = self.vector_db.search(
                query=question,
                k=top_k,
                score_threshold=0.1  # Lowered threshold for better recall
            )
            
            if not search_results:
                return {
                    "answer": "I don't have enough information in the provided documents to answer your question.",
                    "confidence": 0.0,
                    "sources": [],
                    "query": question,
                    "model_info": {
                        "model": model_name or "no_model_used",
                        "tokens_used": 0
                    }
                }
            
            # Extract context from search results
            context = [result["content"] for result in search_results]
            
            # Generate answer using LLM
            if hasattr(llm, 'generate_response') and 'model_name' in llm.generate_response.__code__.co_varnames:
                # New CachedModelLLM supports model_name parameter
                llm_response = llm.generate_response(question, context, model_name)
            else:
                # Fallback for older LLM implementations
                llm_response = llm.generate_response(question, context)
            
            # Prepare sources
            sources = []
            for result in search_results:
                sources.append({
                    "content": result["content"][:200] + "...",
                    "score": result["score"],
                    "metadata": result["metadata"]
                })
            
            # Log query for monitoring
            query_log = {
                "question": question,
                "answer": llm_response["answer"],
                "confidence": llm_response["confidence"],
                "sources_count": len(sources),
                "model": model_name or llm_response.get("model", "unknown")
            }
            self.query_history.append(query_log)
            
            if settings.enable_monitoring:
                self._log_to_monitoring(query_log)
            
            return {
                "answer": llm_response["answer"],
                "confidence": llm_response["confidence"],
                "sources": sources,
                "query": question,
                "model_info": {
                    "model": model_name or llm_response.get("model", "unknown"),
                    "tokens_used": llm_response.get("tokens_used", 0)
                }
            }
            
        except Exception as e:
            app_logger.error(f"Error processing query: {str(e)}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            stats = self.vector_db.get_stats()
            stats["query_count"] = len(self.query_history)
            
            # Calculate average confidence
            if self.query_history:
                avg_confidence = sum(q["confidence"] for q in self.query_history) / len(self.query_history)
                stats["average_confidence"] = avg_confidence
            else:
                stats["average_confidence"] = 0.0
            
            return stats
            
        except Exception as e:
            app_logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base."""
        try:
            self.vector_db.clear()
            self.query_history = []
            app_logger.info("Knowledge base cleared")
            
        except Exception as e:
            app_logger.error(f"Error clearing knowledge base: {str(e)}")
            raise
    
    def get_recent_queries(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent query history."""
        return self.query_history[-limit:]
    
    def _log_to_monitoring(self, query_log: Dict[str, Any]) -> None:
        """Log query to monitoring system (WandB)."""
        try:
            if settings.wandb_api_key:
                import wandb
                
                # Initialize wandb if not already done
                if not wandb.run:
                    # Set the API key in environment first
                    import os
                    os.environ["WANDB_API_KEY"] = settings.wandb_api_key
                    
                    wandb.init(
                        project="legal-document-analyzer",
                        name=f"rag-session-{datetime.now().strftime('%Y%m%d-%H%M')}"
                    )
                
                # Log metrics
                wandb.log({
                    "query_confidence": query_log["confidence"],
                    "sources_count": query_log["sources_count"],
                    "query_length": len(query_log["question"]),
                    "answer_length": len(query_log["answer"]),
                    "model": query_log["model"]
                })
                
        except Exception as e:
            app_logger.warning(f"Failed to log to monitoring: {str(e)}")
            # Continue without monitoring rather than failing


# Global RAG pipeline instance
rag_pipeline = RAGPipeline()

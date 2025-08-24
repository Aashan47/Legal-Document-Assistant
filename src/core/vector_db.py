"""
Vector database management using FAISS for semantic search.
"""
import os
import pickle
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from langchain.schema import Document
from src.core.config import settings
from src.utils.logging import app_logger


class VectorDatabase:
    """FAISS-based vector database for document embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for cosine similarity
        self.documents = []  # Store document metadata
        self.embeddings = []  # Store embeddings for reconstruction
        
        # Load existing index if available
        self._load_index()
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector database."""
        try:
            if not documents:
                return
            
            # Extract text content
            texts = [doc.page_content for doc in documents]
            
            # Generate embeddings
            app_logger.info(f"Generating embeddings for {len(texts)} documents...")
            embeddings = self.embedding_model.encode(texts, convert_to_numpy=True)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to FAISS index
            self.index.add(embeddings)
            
            # Store documents and embeddings
            self.documents.extend(documents)
            self.embeddings.extend(embeddings.tolist())
            
            app_logger.info(f"Added {len(documents)} documents to vector database")
            
            # Save updated index
            self._save_index()
            
        except Exception as e:
            app_logger.error(f"Error adding documents to vector database: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar documents given a query."""
        try:
            if self.index.ntotal == 0:
                app_logger.warning("Vector database is empty")
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)
            
            # Search in FAISS index
            scores, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx != -1 and score >= score_threshold:  # Valid index and above threshold
                    result = {
                        "document": self.documents[idx],
                        "score": float(score),
                        "content": self.documents[idx].page_content,
                        "metadata": self.documents[idx].metadata
                    }
                    results.append(result)
            
            app_logger.info(f"Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            app_logger.error(f"Error searching vector database: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        return {
            "total_documents": len(self.documents),
            "embedding_dimension": self.embedding_dim,
            "model_name": self.model_name,
            "index_size": self.index.ntotal
        }
    
    def clear(self) -> None:
        """Clear all documents from the database."""
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents = []
        self.embeddings = []
        self._save_index()
        app_logger.info("Vector database cleared")
    
    def _save_index(self) -> None:
        """Save the FAISS index and metadata to disk."""
        try:
            os.makedirs(settings.vector_db_path, exist_ok=True)
            
            # Save FAISS index
            index_path = os.path.join(settings.vector_db_path, "faiss.index")
            faiss.write_index(self.index, index_path)
            
            # Save documents metadata
            docs_path = os.path.join(settings.vector_db_path, "documents.pkl")
            with open(docs_path, 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save embeddings
            embeddings_path = os.path.join(settings.vector_db_path, "embeddings.pkl")
            with open(embeddings_path, 'wb') as f:
                pickle.dump(self.embeddings, f)
            
            app_logger.debug("Vector database saved to disk")
            
        except Exception as e:
            app_logger.error(f"Error saving vector database: {str(e)}")
    
    def _load_index(self) -> None:
        """Load the FAISS index and metadata from disk."""
        try:
            index_path = os.path.join(settings.vector_db_path, "faiss.index")
            docs_path = os.path.join(settings.vector_db_path, "documents.pkl")
            embeddings_path = os.path.join(settings.vector_db_path, "embeddings.pkl")
            
            if all(os.path.exists(path) for path in [index_path, docs_path, embeddings_path]):
                # Load FAISS index
                self.index = faiss.read_index(index_path)
                
                # Load documents
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load embeddings
                with open(embeddings_path, 'rb') as f:
                    self.embeddings = pickle.load(f)
                
                app_logger.info(f"Loaded vector database with {len(self.documents)} documents")
            else:
                app_logger.info("No existing vector database found, starting fresh")
                
        except Exception as e:
            app_logger.warning(f"Error loading vector database: {str(e)}")
            # Initialize empty database on error
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.documents = []
            self.embeddings = []

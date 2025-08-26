"""
Vector database management using ChromaDB for semantic search.
"""
import os
import uuid
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
try:
    from src.core.config import settings
except ImportError:
    from src.core.cloud_config import settings
from src.utils.logging import app_logger


class VectorDatabase:
    """ChromaDB-based vector database for document embeddings."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embedding_model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB client
        chromadb_path = os.path.join(settings.vector_db_path, "chromadb")
        os.makedirs(chromadb_path, exist_ok=True)
        
        # Configure ChromaDB with persistent storage
        self.chroma_client = chromadb.PersistentClient(
            path=chromadb_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection_name = "legal_documents"
        try:
            self.collection = self.chroma_client.get_collection(
                name=self.collection_name
            )
            app_logger.info(f"Loaded existing ChromaDB collection with {self.collection.count()} documents")
        except Exception:
            # Create new collection if it doesn't exist
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal document embeddings"}
            )
            app_logger.info("Created new ChromaDB collection")
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector database."""
        try:
            if not documents:
                return
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for i, doc in enumerate(documents):
                # Generate unique ID for each document chunk
                doc_id = str(uuid.uuid4())
                ids.append(doc_id)
                texts.append(doc.page_content)
                
                # Prepare metadata (ChromaDB requires string values)
                metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)
                    else:
                        metadata[key] = str(value)
                
                # Add chunk index for tracking
                metadata["chunk_index"] = str(i)
                metadatas.append(metadata)
            
            app_logger.info(f"Adding {len(documents)} documents to ChromaDB...")
            
            # Add documents to collection (ChromaDB will automatically generate embeddings)
            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
            
            app_logger.info(f"Successfully added {len(documents)} documents to vector database")
            
        except Exception as e:
            app_logger.error(f"Error adding documents to vector database: {str(e)}")
            raise
    
    def search(self, query: str, k: int = 5, score_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Search for similar documents given a query."""
        try:
            # Check if collection has documents
            collection_count = self.collection.count()
            if collection_count == 0:
                app_logger.warning("Vector database is empty")
                return []
            
            app_logger.info(f"Searching ChromaDB collection with {collection_count} documents")
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=min(k, collection_count),
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc_text, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    # Lower distance = higher similarity
                    similarity_score = 1.0 - distance
                    
                    if similarity_score >= score_threshold:
                        # Create a Document object for compatibility
                        document = Document(
                            page_content=doc_text,
                            metadata=metadata
                        )
                        
                        result = {
                            "document": document,
                            "score": float(similarity_score),
                            "content": doc_text,
                            "metadata": metadata
                        }
                        formatted_results.append(result)
            
            app_logger.info(f"Found {len(formatted_results)} relevant documents for query")
            return formatted_results
            
        except Exception as e:
            app_logger.error(f"Error searching vector database: {str(e)}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            collection_count = self.collection.count()
            return {
                "total_documents": collection_count,
                "embedding_dimension": self.embedding_model.get_sentence_embedding_dimension(),
                "model_name": self.model_name,
                "index_size": collection_count,
                "collection_name": self.collection_name
            }
        except Exception as e:
            app_logger.error(f"Error getting database stats: {str(e)}")
            return {
                "total_documents": 0,
                "embedding_dimension": 0,
                "model_name": self.model_name,
                "index_size": 0,
                "collection_name": self.collection_name
            }
    
    def clear(self) -> None:
        """Clear all documents from the database."""
        try:
            # Delete the existing collection
            try:
                self.chroma_client.delete_collection(name=self.collection_name)
                app_logger.info("Deleted existing ChromaDB collection")
            except Exception:
                pass  # Collection might not exist
            
            # Create a new empty collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"description": "Legal document embeddings"}
            )
            
            app_logger.info("Vector database cleared")
            
        except Exception as e:
            app_logger.error(f"Error clearing vector database: {str(e)}")
            raise

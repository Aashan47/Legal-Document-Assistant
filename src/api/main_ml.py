"""
FastAPI backend with full ML features for Legal Document Analyzer
"""
import os
import sys
import tempfile
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import our ML components
from src.core.rag import rag_pipeline
from src.core.config import settings
from src.utils.logging import app_logger

# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Analyzer API",
    description="ML-powered legal document analysis using RAG pipeline",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    max_results: int = 5

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    query: str
    model_info: Dict[str, Any]
    processing_time: float

class UploadResponse(BaseModel):
    message: str
    filename: str
    success: bool
    chunks_processed: int

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    ml_status: Dict[str, Any]

# Global state
documents_processed = []
query_count = 0

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Legal Document Analyzer API with Full ML Features",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process a legal document"""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Check file size (limit to 10MB)
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(content)
            temp_file_path = temp_file.name
        
        try:
            # Process document using RAG pipeline
            result = rag_pipeline.add_documents([temp_file_path])
            
            if result and result.get("success"):
                chunks_processed = result.get("total_chunks", 0)
                documents_processed.append({
                    "filename": file.filename,
                    "upload_time": datetime.now(),
                    "chunks": chunks_processed,
                    "size": len(content)
                })
                
                app_logger.info(f"Successfully processed {file.filename} with {chunks_processed} chunks")
                
                return UploadResponse(
                    message=f"Document processed successfully. {chunks_processed} chunks created.",
                    filename=file.filename,
                    success=True,
                    chunks_processed=chunks_processed
                )
            else:
                error_msg = "Document processing failed"
                if result and "failed" in result:
                    failed_files = result["failed"]
                    if failed_files:
                        error_msg = f"Processing failed: {failed_files[0].get('error', 'Unknown error')}"
                raise HTTPException(status_code=500, detail=error_msg)
                
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                app_logger.warning(f"Failed to delete temp file: {e}")
    
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the processed documents"""
    global query_count
    try:
        query_count += 1
        start_time = datetime.now()
        
        app_logger.info(f"Processing query #{query_count}: {request.question[:100]}...")
        
        # Query the RAG pipeline
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.max_results
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            sources=result["sources"],
            query=result["query"],
            model_info=result.get("model_info", {"model": "unknown", "tokens_used": 0}),
            processing_time=processing_time
        )
        
        app_logger.info(f"Query #{query_count} completed in {processing_time:.2f}s with confidence {result['confidence']:.2f}")
        
        return response
        
    except Exception as e:
        app_logger.error(f"Query error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with ML status"""
    try:
        # Get RAG pipeline stats
        stats = rag_pipeline.get_database_stats()
        
        ml_status = {
            "vector_db_available": stats.get("ml_enabled", False),
            "total_documents": stats.get("total_documents", 0),
            "embedding_model": stats.get("embedding_model", "unknown"),
            "llm_model": getattr(settings, 'default_llm', 'unknown'),
            "average_confidence": stats.get("average_confidence", 0.0)
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="2.0.0",
            ml_status=ml_status
        )
        
    except Exception as e:
        app_logger.error(f"Health check error: {str(e)}")
        return HealthResponse(
            status="degraded",
            timestamp=datetime.now(),
            version="2.0.0",
            ml_status={"error": str(e)}
        )

@app.get("/stats")
async def get_statistics():
    """Get detailed statistics about the system"""
    try:
        db_stats = rag_pipeline.get_database_stats()
        recent_queries = rag_pipeline.get_recent_queries(limit=5)
        
        return {
            "database": db_stats,
            "api": {
                "total_queries": query_count,
                "documents_uploaded": len(documents_processed),
                "recent_uploads": documents_processed[-5:] if documents_processed else []
            },
            "recent_queries": recent_queries,
            "config": {
                "max_chunk_size": getattr(settings, 'chunk_size', 1000),
                "chunk_overlap": getattr(settings, 'chunk_overlap', 200),
                "default_llm": getattr(settings, 'default_llm', 'unknown'),
                "embedding_model": getattr(settings, 'embedding_model', 'all-MiniLM-L6-v2')
            }
        }
        
    except Exception as e:
        app_logger.error(f"Stats error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@app.delete("/documents")
async def clear_documents():
    """Clear all processed documents"""
    try:
        rag_pipeline.clear_knowledge_base()
        documents_processed.clear()
        
        app_logger.info("All documents cleared from the system")
        
        return {"message": "All documents have been cleared successfully"}
        
    except Exception as e:
        app_logger.error(f"Clear documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to clear documents: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all processed documents"""
    try:
        stats = rag_pipeline.get_database_stats()
        
        return {
            "total_documents": len(documents_processed),
            "total_chunks": stats.get("total_documents", 0),
            "documents": documents_processed,
            "ml_enabled": stats.get("ml_enabled", False)
        }
        
    except Exception as e:
        app_logger.error(f"List documents error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

# Example legal questions endpoint
@app.get("/examples")
async def get_example_questions():
    """Get example legal questions for testing"""
    return {
        "examples": [
            "What are the key terms and conditions in this contract?",
            "What are the termination clauses mentioned in the agreement?",
            "Are there any liability limitations specified?",
            "What are the payment terms and conditions?",
            "What intellectual property rights are mentioned?",
            "Are there any non-disclosure or confidentiality clauses?",
            "What are the dispute resolution mechanisms?",
            "What are the governing law and jurisdiction clauses?",
            "Are there any indemnification provisions?",
            "What are the force majeure clauses?"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    app_logger.info("Starting Legal Document Analyzer API with full ML features")
    
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )

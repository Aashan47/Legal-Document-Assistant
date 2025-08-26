"""
FastAPI backend for the Legal Document Analyzer.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import shutil
from pathlib import Path
import uvicorn
from pydantic import BaseModel

from src.core.rag import rag_pipeline
from src.core.config import settings
from src.utils.logging import app_logger


# Pydantic models for API
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5
    model_name: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    query: str
    model_info: Dict[str, Any]


class StatsResponse(BaseModel):
    total_documents: int
    embedding_dimension: int
    model_name: str
    index_size: int
    query_count: int
    average_confidence: float


class UploadResponse(BaseModel):
    success: List[Dict[str, Any]]
    failed: List[Dict[str, Any]]
    total_chunks: int


# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Analyzer API",
    description="RAG-based API for legal document analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    app_logger.info("Starting Legal Document Analyzer API")
    
    # Ensure upload directory exists
    os.makedirs(settings.upload_path, exist_ok=True)
    
    # Log configuration
    app_logger.info(f"Vector DB path: {settings.vector_db_path}")
    app_logger.info(f"Upload path: {settings.upload_path}")
    app_logger.info(f"Default LLM: {settings.default_llm}")
    
    # Initialize model cache in background
    try:
        from src.models.model_manager import get_model_manager
        manager = get_model_manager()
        cache_info = manager.get_cache_info()
        app_logger.info(f"Model cache status: {cache_info['total_cached']}/{cache_info['total_available']} models cached")
        
        if cache_info['total_cached'] == 0:
            app_logger.info("No models cached. Consider running 'python manage_models.py preload' to pre-cache models for faster switching.")
        else:
            app_logger.info("Model cache initialized successfully")
            
    except Exception as e:
        app_logger.warning(f"Model cache initialization warning: {e}")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Legal Document Analyzer API",
        "version": "1.0.0",
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        stats = rag_pipeline.get_database_stats()
        return {
            "status": "healthy",
            "database_connected": True,
            "documents_loaded": stats.get("total_documents", 0)
        }
    except Exception as e:
        app_logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Service unhealthy")


@app.post("/upload", response_model=UploadResponse)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...)
):
    """Upload and process legal documents."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        # Save uploaded files
        file_paths = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are supported. Received: {file.filename}"
                )
            
            # Create unique filename
            file_path = os.path.join(settings.upload_path, file.filename)
            counter = 1
            while os.path.exists(file_path):
                name, ext = os.path.splitext(file.filename)
                file_path = os.path.join(settings.upload_path, f"{name}_{counter}{ext}")
                counter += 1
            
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            file_paths.append(file_path)
        
        # Process documents in background
        background_tasks.add_task(process_documents_background, file_paths)
        
        return JSONResponse(
            content={
                "message": f"Uploaded {len(files)} files. Processing in background.",
                "files": [os.path.basename(path) for path in file_paths]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_documents_background(file_paths: List[str]):
    """Background task to process uploaded documents."""
    try:
        app_logger.info(f"Processing {len(file_paths)} documents in background")
        result = rag_pipeline.add_documents(file_paths)
        app_logger.info(f"Background processing completed: {result['total_chunks']} chunks added")
        
    except Exception as e:
        app_logger.error(f"Background processing failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the legal document knowledge base."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Log the request
        app_logger.info(f"Received query request: {request.question[:100]}...")
        
        # Check database stats before query
        stats = rag_pipeline.get_database_stats()
        app_logger.info(f"Database stats before query: {stats['total_documents']} documents")
        
        if stats['total_documents'] == 0:
            app_logger.warning("No documents in database")
            return QueryResponse(
                answer="I don't have any documents in my knowledge base. Please upload some legal documents first.",
                confidence=0.0,
                sources=[],
                query=request.question,
                model_info={
                    "model": "no_documents",
                    "tokens_used": 0
                }
            )
        
        result = rag_pipeline.query(
            question=request.question,
            top_k=request.top_k,
            model_name=request.model_name
        )
        
        # Log the result
        app_logger.info(f"Query completed: {len(result.get('sources', []))} sources found, confidence: {result.get('confidence', 0):.2f}")
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        app_logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get knowledge base statistics."""
    try:
        stats = rag_pipeline.get_database_stats()
        return StatsResponse(**stats)
        
    except Exception as e:
        app_logger.error(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/queries/recent")
async def get_recent_queries(limit: int = 10):
    """Get recent query history."""
    try:
        queries = rag_pipeline.get_recent_queries(limit=limit)
        return {"queries": queries}
        
    except Exception as e:
        app_logger.error(f"Error getting recent queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/clear")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base."""
    try:
        rag_pipeline.clear_knowledge_base()
        
        # Also remove uploaded files
        for file_path in Path(settings.upload_path).glob("*.pdf"):
            file_path.unlink()
        
        return {"message": "Knowledge base cleared successfully"}
        
    except Exception as e:
        app_logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List uploaded documents."""
    try:
        files = []
        upload_dir = Path(settings.upload_path)
        
        if upload_dir.exists():
            for file_path in upload_dir.glob("*.pdf"):
                stat = file_path.stat()
                files.append({
                    "filename": file_path.name,
                    "size": stat.st_size,
                    "uploaded": stat.st_mtime
                })
        
        return {"documents": files}
        
    except Exception as e:
        app_logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    app_logger.info(f"Starting server on {settings.api_host}:{settings.api_port}")
    uvicorn.run(
        "src.api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )

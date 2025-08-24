"""
Minimal FastAPI app for Legal Document Analyzer - Basic Version
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import os
import shutil
from pathlib import Path
from pydantic import BaseModel
import json

# Simple in-memory storage for this demo
documents_store = []
queries_store = []

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 5

class QueryResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]
    query: str

# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Analyzer API - Basic Version",
    description="Basic version for testing and development",
    version="1.0.0-basic"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    print("🚀 Starting Legal Document Analyzer API - Basic Version")
    
    # Ensure upload directory exists
    os.makedirs("./data/uploads", exist_ok=True)

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Legal Document Analyzer API - Basic Version",
        "version": "1.0.0-basic",
        "status": "running",
        "note": "This is a simplified version for testing. Full ML features will be added once dependencies are resolved."
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database_connected": True,
        "documents_loaded": len(documents_store),
        "version": "basic"
    }

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload and process legal documents - Basic version."""
    try:
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        uploaded_files = []
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Only PDF files are supported. Received: {file.filename}"
                )
            
            # Save file
            file_path = f"./data/uploads/{file.filename}"
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Store basic info (in real version, this would be processed)
            doc_info = {
                "filename": file.filename,
                "path": file_path,
                "size": os.path.getsize(file_path),
                "content": f"[Mock content for {file.filename} - Full text processing will be available in complete version]"
            }
            documents_store.append(doc_info)
            uploaded_files.append(file.filename)
        
        return {
            "message": f"Uploaded {len(files)} files successfully",
            "files": uploaded_files,
            "note": "Files saved. Full text processing will be available in complete version."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error uploading documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Query the legal document knowledge base - Basic version."""
    try:
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Mock response for testing (in real version, this would use ML models)
        mock_answer = f"""Based on the uploaded documents, I can provide analysis for: "{request.question}"

[This is a mock response from the basic version]

Key points to consider:
• Contract terms and conditions
• Legal obligations and responsibilities  
• Potential risks and limitations
• Compliance requirements

Note: This is a simplified response. The full version will provide detailed analysis using advanced NLP models and semantic search."""

        # Mock sources
        mock_sources = []
        for i, doc in enumerate(documents_store[:3]):  # Use first 3 docs
            mock_sources.append({
                "content": f"Relevant section from {doc['filename']}: [Mock content - full text analysis coming soon]",
                "score": 0.85 - (i * 0.1),
                "metadata": {
                    "source": doc["filename"],
                    "chunk_id": i,
                    "file_path": doc["path"]
                }
            })
        
        # Store query
        query_info = {
            "question": request.question,
            "answer": mock_answer,
            "confidence": 0.75,  # Mock confidence
            "sources_count": len(mock_sources),
            "timestamp": "2025-08-24T00:00:00"
        }
        queries_store.append(query_info)
        
        return QueryResponse(
            answer=mock_answer,
            confidence=0.75,
            sources=mock_sources,
            query=request.question
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get knowledge base statistics - Basic version."""
    try:
        return {
            "total_documents": len(documents_store),
            "query_count": len(queries_store),
            "average_confidence": 0.75,
            "version": "basic",
            "note": "Statistics from basic version. Full analytics available in complete version."
        }
        
    except Exception as e:
        print(f"Error getting stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/queries/recent")
async def get_recent_queries(limit: int = 10):
    """Get recent query history."""
    try:
        recent = queries_store[-limit:] if queries_store else []
        return {"queries": recent}
        
    except Exception as e:
        print(f"Error getting recent queries: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear")
async def clear_knowledge_base():
    """Clear all documents from the knowledge base."""
    try:
        global documents_store, queries_store
        documents_store = []
        queries_store = []
        
        # Remove uploaded files
        upload_dir = Path("./data/uploads")
        if upload_dir.exists():
            for file_path in upload_dir.glob("*.pdf"):
                file_path.unlink()
        
        return {"message": "Knowledge base cleared successfully"}
        
    except Exception as e:
        print(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List uploaded documents."""
    try:
        return {"documents": documents_store}
        
    except Exception as e:
        print(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Legal Document Analyzer - Basic Version")
    print("📍 API will be available at: http://localhost:8000")
    print("📖 Documentation at: http://localhost:8000/docs")
    uvicorn.run(app, host="localhost", port=8000)

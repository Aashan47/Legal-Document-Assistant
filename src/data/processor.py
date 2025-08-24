"""
Document processing utilities for the Legal Document Analyzer.
"""
import os
import hashlib
from typing import List, Dict, Optional
from pathlib import Path
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from src.core.config import settings
from src.utils.logging import app_logger


class DocumentProcessor:
    """Handles document processing and chunking."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text content from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            app_logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            raise
    
    def chunk_document(self, text: str, source: str) -> List[Document]:
        """Split document text into chunks."""
        try:
            chunks = self.text_splitter.split_text(text)
            documents = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": source,
                    "chunk_id": i,
                    "total_chunks": len(chunks),
                    "chunk_size": len(chunk)
                }
                documents.append(Document(page_content=chunk, metadata=metadata))
            
            app_logger.info(f"Created {len(documents)} chunks from document {source}")
            return documents
            
        except Exception as e:
            app_logger.error(f"Error chunking document: {str(e)}")
            raise
    
    def process_uploaded_file(self, file_path: str) -> List[Document]:
        """Process an uploaded file and return document chunks."""
        try:
            # Extract text from PDF
            text = self.extract_text_from_pdf(file_path)
            
            # Generate unique source identifier
            file_hash = self._generate_file_hash(file_path)
            source = f"{Path(file_path).stem}_{file_hash}"
            
            # Chunk the document
            documents = self.chunk_document(text, source)
            
            app_logger.info(f"Successfully processed file {file_path}")
            return documents
            
        except Exception as e:
            app_logger.error(f"Error processing file {file_path}: {str(e)}")
            raise
    
    def _generate_file_hash(self, file_path: str) -> str:
        """Generate a hash for the file to create unique identifiers."""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:8]
    
    def validate_file(self, file_path: str) -> bool:
        """Validate if the file is a supported PDF."""
        try:
            if not file_path.lower().endswith('.pdf'):
                return False
            
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                # Try to read first page to validate
                if len(reader.pages) > 0:
                    reader.pages[0].extract_text()
                return True
                
        except Exception as e:
            app_logger.warning(f"File validation failed for {file_path}: {str(e)}")
            return False

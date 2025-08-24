"""
Test suite for the Legal Document Analyzer.
"""
import unittest
import os
import tempfile
from pathlib import Path
import sys

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.data.processor import DocumentProcessor
from src.core.vector_db import VectorDatabase
from src.models.llm import LLMFactory
from src.core.rag import RAGPipeline


class TestDocumentProcessor(unittest.TestCase):
    """Test document processing functionality."""
    
    def setUp(self):
        self.processor = DocumentProcessor()
    
    def test_chunk_document(self):
        """Test document chunking."""
        text = "This is a test document. " * 100
        source = "test_doc"
        
        documents = self.processor.chunk_document(text, source)
        
        self.assertGreater(len(documents), 0)
        self.assertEqual(documents[0].metadata["source"], source)
    
    def test_validate_file(self):
        """Test file validation."""
        # Test non-PDF file
        self.assertFalse(self.processor.validate_file("test.txt"))
        
        # Test non-existent file
        self.assertFalse(self.processor.validate_file("nonexistent.pdf"))


class TestVectorDatabase(unittest.TestCase):
    """Test vector database functionality."""
    
    def setUp(self):
        # Use temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.vector_db = VectorDatabase()
    
    def test_get_stats(self):
        """Test database statistics."""
        stats = self.vector_db.get_stats()
        
        self.assertIn("total_documents", stats)
        self.assertIn("embedding_dimension", stats)
        self.assertIn("model_name", stats)
    
    def test_search_empty_db(self):
        """Test searching empty database."""
        results = self.vector_db.search("test query")
        self.assertEqual(len(results), 0)


class TestRAGPipeline(unittest.TestCase):
    """Test RAG pipeline functionality."""
    
    def setUp(self):
        self.rag = RAGPipeline()
    
    def test_get_database_stats(self):
        """Test getting database statistics."""
        stats = self.rag.get_database_stats()
        self.assertIsInstance(stats, dict)
    
    def test_query_empty_knowledge_base(self):
        """Test querying empty knowledge base."""
        result = self.rag.query("What is this document about?")
        
        self.assertIn("answer", result)
        self.assertIn("confidence", result)
        self.assertEqual(result["confidence"], 0.0)


class TestLLMFactory(unittest.TestCase):
    """Test LLM factory functionality."""
    
    def test_create_llm_invalid_type(self):
        """Test creating LLM with invalid type."""
        with self.assertRaises(ValueError):
            LLMFactory.create_llm("invalid_llm_type")


def create_sample_pdf():
    """Create a sample PDF for testing (placeholder)."""
    # This would require reportlab or similar
    # For now, just return a path that would fail validation
    return "sample.pdf"


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)

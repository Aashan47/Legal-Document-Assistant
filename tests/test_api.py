"""
API tests for the Legal Document Analyzer.
"""
import unittest
import requests
import time
import threading
import subprocess
import sys
from pathlib import Path


class TestAPI(unittest.TestCase):
    """Test API endpoints."""
    
    @classmethod
    def setUpClass(cls):
        """Start API server for testing."""
        cls.api_url = "http://localhost:8000"
        
        # Try to start API server
        try:
            # Check if server is already running
            response = requests.get(f"{cls.api_url}/health", timeout=2)
            if response.status_code == 200:
                cls.server_process = None
                return
        except:
            pass
        
        # Start server if not running
        try:
            cls.server_process = subprocess.Popen([
                sys.executable, "-m", "uvicorn",
                "src.api.main:app",
                "--host", "localhost",
                "--port", "8000"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Wait for server to start
            time.sleep(5)
        except Exception as e:
            cls.server_process = None
            raise unittest.SkipTest(f"Could not start API server: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Stop API server after testing."""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = requests.get(f"{self.api_url}/")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("message", data)
        self.assertIn("version", data)
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = requests.get(f"{self.api_url}/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("status", data)
        self.assertEqual(data["status"], "healthy")
    
    def test_stats_endpoint(self):
        """Test stats endpoint."""
        response = requests.get(f"{self.api_url}/stats")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("total_documents", data)
        self.assertIn("embedding_dimension", data)
    
    def test_query_endpoint(self):
        """Test query endpoint."""
        query_data = {
            "question": "What is this document about?",
            "top_k": 5
        }
        
        response = requests.post(f"{self.api_url}/query", json=query_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("confidence", data)
        self.assertIn("sources", data)
    
    def test_query_empty_question(self):
        """Test query with empty question."""
        query_data = {"question": ""}
        
        response = requests.post(f"{self.api_url}/query", json=query_data)
        self.assertEqual(response.status_code, 400)
    
    def test_recent_queries_endpoint(self):
        """Test recent queries endpoint."""
        response = requests.get(f"{self.api_url}/queries/recent")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("queries", data)
        self.assertIsInstance(data["queries"], list)


if __name__ == '__main__':
    unittest.main(verbosity=2)

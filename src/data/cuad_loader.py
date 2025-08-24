"""
CUAD (Contract Understanding Atticus Dataset) data loader.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import load_dataset
from pathlib import Path

from src.core.config import settings
from src.utils.logging import app_logger
from src.core.rag import rag_pipeline


class CUADLoader:
    """Loader for the CUAD dataset."""
    
    def __init__(self):
        self.dataset_path = settings.cuad_data_path
        self.dataset = None
    
    def download_cuad_dataset(self) -> bool:
        """Download the CUAD dataset from Hugging Face."""
        try:
            app_logger.info("Downloading CUAD dataset...")
            
            # Load dataset from Hugging Face
            dataset = load_dataset("cuad", split="train")
            
            # Save dataset locally
            os.makedirs(self.dataset_path, exist_ok=True)
            dataset.save_to_disk(self.dataset_path)
            
            app_logger.info(f"CUAD dataset downloaded to {self.dataset_path}")
            return True
            
        except Exception as e:
            app_logger.error(f"Error downloading CUAD dataset: {str(e)}")
            return False
    
    def load_cuad_dataset(self) -> Optional[Any]:
        """Load the CUAD dataset."""
        try:
            from datasets import load_from_disk
            
            if not os.path.exists(self.dataset_path):
                app_logger.info("CUAD dataset not found locally. Downloading...")
                if not self.download_cuad_dataset():
                    return None
            
            self.dataset = load_from_disk(self.dataset_path)
            app_logger.info(f"Loaded CUAD dataset with {len(self.dataset)} examples")
            return self.dataset
            
        except Exception as e:
            app_logger.error(f"Error loading CUAD dataset: {str(e)}")
            return None
    
    def extract_sample_contracts(self, num_samples: int = 10) -> List[Dict[str, Any]]:
        """Extract sample contracts from CUAD for testing."""
        try:
            if self.dataset is None:
                self.load_cuad_dataset()
            
            if self.dataset is None:
                return []
            
            samples = []
            for i in range(min(num_samples, len(self.dataset))):
                example = self.dataset[i]
                samples.append({
                    "title": example.get("title", f"Contract_{i}"),
                    "context": example.get("context", ""),
                    "questions": example.get("questions", []),
                    "answers": example.get("answers", [])
                })
            
            app_logger.info(f"Extracted {len(samples)} sample contracts from CUAD")
            return samples
            
        except Exception as e:
            app_logger.error(f"Error extracting CUAD samples: {str(e)}")
            return []
    
    def create_test_documents(self, num_samples: int = 5) -> List[str]:
        """Create test PDF-like documents from CUAD samples."""
        try:
            samples = self.extract_sample_contracts(num_samples)
            file_paths = []
            
            for i, sample in enumerate(samples):
                # Create a text file (simulating processed PDF content)
                filename = f"cuad_sample_{i}_{sample['title'].replace(' ', '_')}.txt"
                file_path = os.path.join(self.dataset_path, filename)
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(f"LEGAL CONTRACT: {sample['title']}\n\n")
                    f.write(sample['context'])
                
                file_paths.append(file_path)
            
            app_logger.info(f"Created {len(file_paths)} test documents from CUAD")
            return file_paths
            
        except Exception as e:
            app_logger.error(f"Error creating test documents: {str(e)}")
            return []
    
    def get_sample_questions(self) -> List[str]:
        """Get sample questions from CUAD for testing."""
        try:
            if self.dataset is None:
                self.load_cuad_dataset()
            
            if self.dataset is None:
                return []
            
            questions = []
            for i in range(min(20, len(self.dataset))):
                example = self.dataset[i]
                example_questions = example.get("questions", [])
                questions.extend(example_questions[:2])  # Take first 2 questions per contract
            
            # Remove duplicates and filter
            unique_questions = list(set(questions))
            filtered_questions = [q for q in unique_questions if len(q) > 10 and len(q) < 200]
            
            return filtered_questions[:15]  # Return top 15 questions
            
        except Exception as e:
            app_logger.error(f"Error getting sample questions: {str(e)}")
            return []
    
    def run_cuad_benchmark(self, num_samples: int = 5) -> Dict[str, Any]:
        """Run a benchmark test using CUAD data."""
        try:
            app_logger.info("Running CUAD benchmark...")
            
            # Load sample contracts into the RAG system
            test_files = self.create_test_documents(num_samples)
            if not test_files:
                return {"error": "Failed to create test documents"}
            
            # Add documents to RAG pipeline
            # Note: We're using text files instead of PDFs for testing
            # In a real implementation, you'd convert these to proper PDFs
            
            # Get sample questions
            questions = self.get_sample_questions()
            if not questions:
                return {"error": "Failed to get sample questions"}
            
            # Run queries
            results = []
            for question in questions[:10]:  # Test with 10 questions
                try:
                    result = rag_pipeline.query(question)
                    results.append({
                        "question": question,
                        "answer": result.get("answer", ""),
                        "confidence": result.get("confidence", 0.0),
                        "sources_count": len(result.get("sources", []))
                    })
                except Exception as e:
                    app_logger.warning(f"Failed to process question: {question[:50]}... Error: {str(e)}")
            
            # Calculate metrics
            if results:
                avg_confidence = sum(r["confidence"] for r in results) / len(results)
                successful_queries = len([r for r in results if r["confidence"] > 0.5])
                
                benchmark_result = {
                    "total_questions": len(results),
                    "successful_queries": successful_queries,
                    "success_rate": successful_queries / len(results),
                    "average_confidence": avg_confidence,
                    "sample_results": results[:5]  # Include first 5 results as examples
                }
            else:
                benchmark_result = {"error": "No successful queries"}
            
            app_logger.info(f"CUAD benchmark completed: {benchmark_result}")
            return benchmark_result
            
        except Exception as e:
            app_logger.error(f"Error running CUAD benchmark: {str(e)}")
            return {"error": str(e)}


# Global CUAD loader instance
cuad_loader = CUADLoader()

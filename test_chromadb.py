"""
Test script for ChromaDB functionality
"""
from src.core.vector_db import VectorDatabase
from langchain.schema import Document

# Initialize vector database
vdb = VectorDatabase()

# Test adding documents
test_docs = [
    Document(page_content='This is a test contract about employment terms.', metadata={'source': 'test1.pdf', 'page': 1}),
    Document(page_content='This document contains privacy policy information.', metadata={'source': 'test2.pdf', 'page': 1})
]

print('Adding test documents...')
vdb.add_documents(test_docs)

print('Updated stats:', vdb.get_stats())

# Test search
print('Testing search...')
results = vdb.search('employment contract', k=2)
print(f'Found {len(results)} results')
for i, result in enumerate(results):
    print(f'Result {i+1}: Score={result["score"]:.3f}, Content={result["content"][:50]}...')

print('ChromaDB functionality test completed successfully!')

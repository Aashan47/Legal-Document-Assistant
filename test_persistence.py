"""
Test script to check if documents persist between queries
"""
import time
from src.core.rag import rag_pipeline

print("=== Document Persistence Test ===")

# Check initial state
stats = rag_pipeline.get_database_stats()
print(f"Initial document count: {stats['total_documents']}")

if stats['total_documents'] == 0:
    print("No documents found. Adding test documents...")
    from langchain.schema import Document
    
    test_docs = [
        Document(
            page_content="This is a test employment contract with termination clauses. The employee can be terminated with 30 days notice.",
            metadata={'source': 'test_contract.pdf', 'page': 1}
        )
    ]
    
    rag_pipeline.add_documents(['test'])  # This will fail, but let's see what happens
    # Instead, let's add directly to vector DB
    rag_pipeline.vector_db.add_documents(test_docs)
    
    # Check again
    stats = rag_pipeline.get_database_stats()
    print(f"After adding documents: {stats['total_documents']}")

# Test first query
print("\n=== First Query ===")
result1 = rag_pipeline.query("What are the termination clauses?", top_k=3)
print(f"Sources found: {len(result1['sources'])}")
print(f"Answer: {result1['answer'][:100]}...")

# Wait a moment
time.sleep(1)

# Check document count again
stats = rag_pipeline.get_database_stats()
print(f"Document count after first query: {stats['total_documents']}")

# Test second query
print("\n=== Second Query ===")
result2 = rag_pipeline.query("What are the employment terms?", top_k=3)
print(f"Sources found: {len(result2['sources'])}")
print(f"Answer: {result2['answer'][:100]}...")

# Final check
stats = rag_pipeline.get_database_stats()
print(f"Final document count: {stats['total_documents']}")

print("\n=== Test Complete ===")
if stats['total_documents'] > 0 and len(result2['sources']) > 0:
    print("✅ Documents are persisting correctly!")
else:
    print("❌ Documents are not persisting between queries!")

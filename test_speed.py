from src.core.rag import rag_pipeline
import time

print('Testing optimized system for speed and quality...')
print('=' * 60)

# Test with a legal question
question = 'What are the termination clauses in the employment agreement?'
print(f'Question: {question}')

start_time = time.time()
result = rag_pipeline.query(question, top_k=3)
end_time = time.time()

print(f'Processing time: {end_time - start_time:.2f} seconds')
print(f'Model: {result["model_info"]["model"]}')
print(f'Answer length: {len(result["answer"])} characters')
print(f'Confidence: {result["confidence"]:.1%}')
print(f'Sources found: {len(result["sources"])}')
print()
print('Answer preview:')
print(result["answer"][:400] + '...' if len(result["answer"]) > 400 else result["answer"])

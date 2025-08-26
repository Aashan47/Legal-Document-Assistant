from src.core.rag import rag_pipeline

print('Testing enhanced system with detailed legal content...')
print('=' * 60)

# Test with a detailed legal question
result = rag_pipeline.query('What are the termination clauses and conditions in the employment agreement?', top_k=5)

print('Results:')
print('- Model:', result['model_info']['model'])
print('- Answer length:', len(result['answer']), 'characters')
print('- Confidence:', f'{result["confidence"]:.2%}')
print('- Sources found:', len(result['sources']))
print()
print('DETAILED ANSWER:')
print('=' * 60)
print(result['answer'])
print()
print('=' * 60)
print('SOURCES USED:')
for i, source in enumerate(result['sources'], 1):
    print(f'{i}. Score: {source["score"]:.3f}')
    print(f'   Content: {source["content"][:150]}...')
    print()

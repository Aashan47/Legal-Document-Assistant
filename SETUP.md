# Legal Document Analyzer - Setup Guide

## Quick Start (5 minutes)

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM recommended
- Internet connection for downloading models

### Option 1: Automated Setup
```bash
# Clone or download the project
cd "Legal Document Analyzer"

# Run the automated setup script
python start.py
```
This script will:
- Check Python version
- Install all dependencies
- Set up environment files
- Start both API and UI

### Option 2: Manual Setup

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2. Configure Environment
```bash
# Copy environment template
cp .env.example .env

# Edit .env with your API keys
notepad .env  # On Windows
# OR
nano .env     # On Linux/Mac
```

#### 3. Start the Services

**Terminal 1 - API Server:**
```bash
python -m uvicorn src.api.main:app --host localhost --port 8000 --reload
```

**Terminal 2 - Streamlit UI:**
```bash
streamlit run src/ui/app.py --server.port 8501
```

#### 4. Access the Application
- **Web UI:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs

## Configuration

### Environment Variables (.env file)
```bash
# Required for OpenAI GPT-4
OPENAI_API_KEY=sk-your-openai-key-here

# Optional for Hugging Face models
HUGGINGFACE_TOKEN=hf_your-token-here

# Optional for monitoring
WANDB_API_KEY=your-wandb-key-here

# System settings (defaults are usually fine)
VECTOR_DB_PATH=./data/vector_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
DEFAULT_LLM=openai
```

### Supported LLM Options

#### 1. OpenAI GPT-4 (Recommended)
- **Pros:** High quality, reliable, fast
- **Cons:** Requires API key, costs money
- **Setup:** Add `OPENAI_API_KEY` to .env

#### 2. Local Llama Models
- **Pros:** Free, private, offline
- **Cons:** Requires model download, slower
- **Setup:** Download model and set path in config

#### 3. Hugging Face Models
- **Pros:** Many options, good performance
- **Cons:** May require GPU for speed
- **Setup:** Add `HUGGINGFACE_TOKEN` to .env

## Usage Examples

### 1. Upload Documents (Web UI)
1. Go to http://localhost:8501
2. Click "Document Upload" tab
3. Drag & drop PDF files
4. Click "Upload & Process Documents"

### 2. Query Documents (Web UI)
1. Go to "Query Documents" tab
2. Type your question: "What are the termination clauses?"
3. Click "Ask Question"
4. Review answer and sources

### 3. Command Line Interface
```bash
# Add documents
python cli.py add-documents contract1.pdf contract2.pdf

# Query the knowledge base
python cli.py query "What are the liability limitations?"

# View statistics
python cli.py stats

# Run benchmark with CUAD dataset
python cli.py benchmark --samples 10

# Clear knowledge base
python cli.py clear
```

### 4. API Usage (Python)
```python
import requests

# Add documents
files = {'files': open('contract.pdf', 'rb')}
response = requests.post('http://localhost:8000/upload', files=files)

# Query
query = {"question": "What are the payment terms?", "top_k": 5}
response = requests.post('http://localhost:8000/query', json=query)
answer = response.json()

print(f"Answer: {answer['answer']}")
print(f"Confidence: {answer['confidence']}")
```

## Sample Questions

Try these example questions with legal contracts:

### Contract Analysis
- "What are the termination clauses in this contract?"
- "What are the liability limitations mentioned?"
- "What are the payment terms and conditions?"
- "Are there any confidentiality obligations?"

### Risk Assessment
- "What are the penalty clauses for breach of contract?"
- "What governing law applies to this agreement?"
- "Are there any indemnification provisions?"
- "What are the force majeure clauses?"

### Compliance
- "What are the intellectual property provisions?"
- "Are there any non-compete clauses?"
- "What are the data protection requirements?"
- "What are the audit and inspection rights?"

## Troubleshooting

### Common Issues

#### 1. API Server Not Starting
```bash
# Check if port 8000 is in use
netstat -an | findstr :8000  # Windows
lsof -i :8000               # Mac/Linux

# Use different port
python -m uvicorn src.api.main:app --port 8001
```

#### 2. Low Confidence Scores
- Upload more relevant documents
- Try rephrasing your question
- Check document quality (clear text, not scanned images)

#### 3. Slow Performance
- Use smaller chunk sizes
- Reduce `top_k` parameter
- Consider using GPU for embeddings

#### 4. Memory Issues
- Reduce batch sizes in processing
- Use smaller embedding models
- Close other applications

### Getting Help

1. **Check logs:** `logs/app.log`
2. **API status:** http://localhost:8000/health
3. **Verbose logging:** Set `LOG_LEVEL=DEBUG` in .env

## Performance Optimization

### 1. Hardware Recommendations
- **CPU:** 4+ cores recommended
- **RAM:** 8GB+ for large document sets
- **Storage:** SSD for better I/O performance

### 2. Configuration Tuning
```bash
# For precision (smaller chunks)
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# For speed (larger chunks)
CHUNK_SIZE=1500
CHUNK_OVERLAP=150

# For better recall
TOP_K=10
CONFIDENCE_THRESHOLD=0.3
```

### 3. Production Deployment
- Use `docker-compose.yml` for containerized deployment
- Set `ENVIRONMENT=production` for production settings
- Configure reverse proxy (nginx) for HTTPS
- Set up monitoring with WandB or similar

## Security Considerations

### 1. API Keys
- Never commit API keys to version control
- Use environment variables
- Rotate keys regularly

### 2. Data Privacy
- Documents are stored locally by default
- Consider encryption for sensitive documents
- Review data retention policies

### 3. Network Security
- Use HTTPS in production
- Configure CORS properly
- Consider VPN for sensitive deployments

## Next Steps

1. **Upload your legal documents** and start querying
2. **Experiment with different question types**
3. **Monitor confidence scores** and tune parameters
4. **Set up monitoring** with WandB for production use
5. **Customize the UI** for your specific needs

## Support

For additional help:
- Check the logs in `logs/app.log`
- Review the API documentation at http://localhost:8000/docs
- Adjust configuration in `.env` file as needed

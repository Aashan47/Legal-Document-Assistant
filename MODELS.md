# Open-Source Models Configuration Guide

## Recommended Open-Source Models for Legal Document Analysis

### 1. Mistral Models (Recommended)
```bash
# In .env file:
HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
```
**Pros:** Excellent for instruction following, good reasoning
**Use case:** Best overall performance for legal Q&A

### 2. Llama 2 Models  
```bash
# In .env file:
HF_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf
```
**Pros:** Strong general capabilities, well-trained on conversations
**Use case:** Good for detailed legal analysis
**Note:** Requires HuggingFace approval for access

### 3. Falcon Models
```bash
# In .env file:
HF_MODEL_NAME=tiiuae/falcon-7b-instruct
```
**Pros:** Fast inference, good multilingual support
**Use case:** Efficient for production deployments

### 4. Code Llama (For Contract Analysis)
```bash
# In .env file:
HF_MODEL_NAME=codellama/CodeLlama-7b-Instruct-hf
```
**Pros:** Good at structured text analysis
**Use case:** Excellent for parsing contract structures

### 5. Zephyr Models (Lightweight)
```bash
# In .env file:
HF_MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
```
**Pros:** Optimized for chat, smaller size
**Use case:** Good balance of performance and resource usage

## Configuration Options

### Option 1: Hugging Face Inference API (Recommended)
```bash
# In .env file:
DEFAULT_LLM=huggingface
HF_USE_INFERENCE_API=true
HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
```

**Advantages:**
- No local GPU required
- Fast startup
- Access to latest models
- Automatic model updates

**Requirements:**
- Hugging Face token (free)
- Internet connection
- Usage limits apply

### Option 2: Local Model (Advanced)
```bash
# In .env file:
DEFAULT_LLM=huggingface
HF_USE_INFERENCE_API=false
HF_MODEL_NAME=microsoft/DialoGPT-medium
```

**Advantages:**
- Complete privacy
- No usage limits
- Offline operation
- Full control

**Requirements:**
- GPU with 8GB+ VRAM (recommended)
- 16GB+ RAM
- Local model download (~3-7GB per model)

## Current Configuration

Your current setup in `.env`:
```bash
DEFAULT_LLM=huggingface
HF_MODEL_NAME=microsoft/DialoGPT-medium
HF_USE_INFERENCE_API=true
```

## Switching Models

To try different models, simply update your `.env` file:

```bash
# For Mistral (recommended for legal analysis)
HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1

# For Llama 2 (requires approval)
HF_MODEL_NAME=meta-llama/Llama-2-7b-chat-hf

# For Falcon (fast and efficient)  
HF_MODEL_NAME=tiiuae/falcon-7b-instruct

# For Zephyr (lightweight)
HF_MODEL_NAME=HuggingFaceH4/zephyr-7b-beta
```

## Performance Comparison

| Model | Size | Speed | Quality | Legal Focus |
|-------|------|-------|---------|-------------|
| Mistral-7B | 7B | Fast | Excellent | ⭐⭐⭐⭐⭐ |
| Llama-2-7B | 7B | Medium | Excellent | ⭐⭐⭐⭐ |
| Falcon-7B | 7B | Fast | Good | ⭐⭐⭐ |
| Zephyr-7B | 7B | Fast | Good | ⭐⭐⭐ |
| DialoGPT | 1.5B | Very Fast | Basic | ⭐⭐ |

## Troubleshooting

### Model Loading Issues
1. **Check HuggingFace token**: Ensure valid token in `.env`
2. **Model access**: Some models require approval (Llama 2)
3. **Memory issues**: Use smaller models or inference API
4. **Network issues**: Check internet connection for API calls

### Performance Optimization
1. **Use GPU**: Set `CUDA_VISIBLE_DEVICES=0` for GPU acceleration
2. **Batch processing**: Process multiple queries together
3. **Model caching**: Models are cached after first use
4. **Reduce context**: Limit document chunks for faster processing

## Getting Started

1. **Update your model** in `.env`:
   ```bash
   HF_MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
   ```

2. **Restart the application**:
   ```bash
   python start.py
   ```

3. **Test with a simple query**:
   - Upload a legal document
   - Ask: "What are the main terms of this agreement?"

4. **Monitor performance**:
   - Check confidence scores
   - Adjust model if needed

## Best Practices

1. **Start with Mistral-7B** for best legal analysis performance
2. **Use inference API** for development and testing
3. **Switch to local models** for production/privacy
4. **Monitor confidence scores** and adjust models accordingly
5. **Keep HuggingFace token secure** and don't commit to git

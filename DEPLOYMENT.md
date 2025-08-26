# Streamlit Cloud Deployment Guide

## 🚀 Deploy to Streamlit Cloud

### Prerequisites
- GitHub repository with your code
- Streamlit Cloud account (free at share.streamlit.io)

### Step 1: Prepare Repository
1. Ensure `streamlit_app.py` is in the root directory
2. Verify `requirements_streamlit.txt` contains all dependencies
3. Make sure `.streamlit/config.toml` is configured
4. Add `packages.txt` for system dependencies

### Step 2: Push to GitHub
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### Step 3: Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click "New app"
4. Select your repository: `Legal-Document-Assistant`
5. Set main file path: `streamlit_app.py`
6. Set requirements file: `requirements_streamlit.txt`
7. Click "Deploy!"

### Step 4: Monitor Deployment
- First deployment takes 5-10 minutes
- Watch build logs for any errors
- App will be available at: `https://[your-app-name].streamlit.app`

## 📋 File Structure for Deployment

```
Legal-Document-Assistant/
├── streamlit_app.py              # Main Streamlit app
├── requirements_streamlit.txt    # Python dependencies
├── packages.txt                  # System dependencies
├── .streamlit/
│   └── config.toml              # Streamlit configuration
├── src/
│   ├── core/
│   │   ├── cloud_config.py      # Cloud-optimized settings
│   │   ├── vector_db.py         # Vector database
│   │   └── rag.py               # RAG pipeline
│   ├── models/
│   │   ├── model_config.py      # Model configurations
│   │   └── llm.py               # LLM implementations
│   ├── data/
│   │   └── processor.py         # Document processing
│   └── utils/
│       └── logging.py           # Logging utilities
└── README.md
```

## ⚠️ Important Notes

### Model Selection for Cloud
- Only FLAN-T5 models are enabled for cloud deployment
- DialoGPT models removed due to compatibility issues
- Models will download on first use (takes time)

### Storage Limitations
- Streamlit Cloud has limited persistent storage
- Vector database resets on app restart
- Users need to re-upload documents after restarts

### Performance Considerations
- First model load takes 2-3 minutes
- Subsequent queries are faster
- Use smaller models (flan-t5-small) for better performance

### Memory Limitations
- Streamlit Cloud has 1GB RAM limit
- Large models may cause memory issues
- Monitor resource usage in app

## 🛠️ Troubleshooting

### Common Issues
1. **Build fails**: Check requirements_streamlit.txt for version conflicts
2. **Model loading errors**: Ensure transformers version compatibility
3. **Memory errors**: Use smaller models or reduce batch sizes
4. **File upload issues**: Check file size limits in config.toml

### Debug Steps
1. Check Streamlit Cloud build logs
2. Test locally with: `streamlit run streamlit_app.py`
3. Verify all imports work
4. Test with small documents first

## 🎯 Optimization Tips

### For Better Performance
1. Use `st.cache_resource` for model loading
2. Implement progressive loading for large models
3. Add loading indicators for better UX
4. Optimize chunk sizes for cloud environment

### For Better User Experience
1. Add clear instructions for first-time users
2. Show model loading progress
3. Implement error handling and retries
4. Add example documents and queries

## 📞 Support
If you encounter issues:
1. Check Streamlit Community Forum
2. Review deployment logs
3. Test locally first
4. Verify all dependencies are included

"""
Simple test script to check imports and start a basic version
"""
import sys
import os

print("Testing imports...")

try:
    import fastapi
    print("✅ FastAPI imported successfully")
except ImportError as e:
    print(f"❌ FastAPI import failed: {e}")

try:
    import streamlit
    print("✅ Streamlit imported successfully")
except ImportError as e:
    print(f"❌ Streamlit import failed: {e}")

try:
    import sentence_transformers
    print("✅ Sentence Transformers imported successfully")
except ImportError as e:
    print(f"❌ Sentence Transformers import failed: {e}")

try:
    import faiss
    print("✅ FAISS imported successfully")
except ImportError as e:
    print(f"❌ FAISS import failed: {e}")
    print("Trying alternative import...")
    try:
        import faiss_cpu as faiss
        print("✅ FAISS-CPU imported successfully")
    except ImportError as e2:
        print(f"❌ FAISS-CPU import also failed: {e2}")

try:
    import transformers
    print("✅ Transformers imported successfully")
except ImportError as e:
    print(f"❌ Transformers import failed: {e}")

print("\nPython path:")
for path in sys.path:
    print(f"  {path}")

print(f"\nCurrent working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")

"""
Startup script for the Legal Document Analyzer.
This script helps users get started quickly with the application.
"""
import os
import sys
import subprocess
import time
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        "fastapi", "uvicorn", "streamlit", "torch", "transformers",
        "sentence-transformers", "faiss-cpu", "langchain", "openai"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                *missing_packages
            ])
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages. Please install manually:")
            print(f"pip install {' '.join(missing_packages)}")
            return False
    
    return True

def setup_environment():
    """Set up environment variables and directories."""
    env_file = Path(".env")
    env_example = Path(".env.example")
    
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        env_file.write_text(env_example.read_text())
        print("✅ .env file created. Please edit it with your API keys.")
    
    # Create necessary directories
    directories = ["data", "logs", "data/uploads", "data/vector_db", "data/cuad"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Directory created: {directory}")

def start_api_server():
    """Start the FastAPI server."""
    print("\n🚀 Starting API server...")
    try:
        # Start API server in background
        process = subprocess.Popen([
            sys.executable, "-m", "uvicorn", 
            "src.api.main:app",
            "--host", "localhost",
            "--port", "8000",
            "--reload"
        ])
        
        # Wait a moment for server to start
        time.sleep(3)
        
        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ API server is running at http://localhost:8000")
                return process
            else:
                print("❌ API server failed to start properly")
                return None
        except:
            print("❌ API server is not responding")
            return None
            
    except Exception as e:
        print(f"❌ Failed to start API server: {str(e)}")
        return None

def start_streamlit_app():
    """Start the Streamlit application."""
    print("\n🎨 Starting Streamlit application...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/ui/app.py",
            "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n👋 Shutting down application...")
    except Exception as e:
        print(f"❌ Failed to start Streamlit app: {str(e)}")

def main():
    """Main startup function."""
    print("🏛️ Legal Document Analyzer - Startup Script")
    print("=" * 50)
    
    # Check system requirements
    if not check_python_version():
        return
    
    # Set up environment
    setup_environment()
    
    # Check and install dependencies
    if not check_dependencies():
        print("❌ Dependency check failed. Please resolve issues and try again.")
        return
    
    print("\n✅ System check completed successfully!")
    print("\nStarting Legal Document Analyzer...")
    
    # Start API server
    api_process = start_api_server()
    
    if api_process:
        try:
            # Start Streamlit app (this will block)
            start_streamlit_app()
        finally:
            # Clean up API server
            if api_process:
                print("\n🛑 Stopping API server...")
                api_process.terminate()
                api_process.wait()
    else:
        print("❌ Cannot start Streamlit app without API server")

if __name__ == "__main__":
    main()

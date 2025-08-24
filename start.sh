#!/bin/bash
# Linux/Mac startup script for the Legal Document Analyzer

echo "====================================================="
echo "    Legal Document Analyzer - Unix Startup"
echo "====================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "Python found: $(python3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo ""
    echo "IMPORTANT: Please edit .env file with your API keys before proceeding"
    echo "Press Enter to continue after editing .env file..."
    read
fi

echo ""
echo "Starting Legal Document Analyzer..."
echo ""
echo "API will be available at: http://localhost:8000"
echo "Web UI will be available at: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    jobs -p | xargs -r kill
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Start API server in background
python -m uvicorn src.api.main:app --host localhost --port 8000 --reload &
API_PID=$!

# Wait a bit for API to start
sleep 5

# Start Streamlit UI (this will block)
python -m streamlit run src/ui/app.py --server.port 8501

# Cleanup
cleanup

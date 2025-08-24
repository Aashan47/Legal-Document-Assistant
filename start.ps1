# PowerShell script to start the Legal Document Analyzer
param(
    [switch]$SkipInstall,
    [switch]$ApiOnly,
    [switch]$UiOnly
)

Write-Host "=====================================================" -ForegroundColor Green
Write-Host "    Legal Document Analyzer - PowerShell Startup" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green

# Check if Python is installed
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Python found: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "ERROR: Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install dependencies unless skipped
if (-not $SkipInstall) {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
}

# Create .env file if it doesn't exist
if (-not (Test-Path ".env")) {
    Write-Host "Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item ".env.example" ".env"
    Write-Host ""
    Write-Host "IMPORTANT: Please edit .env file with your API keys before proceeding" -ForegroundColor Red
    Write-Host "Opening .env file in default editor..." -ForegroundColor Yellow
    Start-Process ".env"
    Read-Host "Press Enter to continue after editing .env file"
}

Write-Host ""
Write-Host "Starting Legal Document Analyzer..." -ForegroundColor Green
Write-Host ""
Write-Host "API will be available at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Web UI will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the application" -ForegroundColor Yellow
Write-Host ""

# Function to start API server
function Start-ApiServer {
    Write-Host "Starting API server..." -ForegroundColor Green
    $apiJob = Start-Job -ScriptBlock {
        Set-Location $using:PWD
        & "venv\Scripts\Activate.ps1"
        python -m uvicorn src.api.main:app --host localhost --port 8000 --reload
    }
    return $apiJob
}

# Function to start Streamlit UI
function Start-StreamlitUI {
    Write-Host "Starting Streamlit UI..." -ForegroundColor Green
    Start-Sleep -Seconds 3  # Wait for API to start
    python -m streamlit run src\ui\app.py --server.port 8501
}

# Start services based on parameters
try {
    if ($ApiOnly) {
        $apiJob = Start-ApiServer
        Write-Host "API server started. Press Ctrl+C to stop..." -ForegroundColor Green
        Wait-Job $apiJob
    } elseif ($UiOnly) {
        Start-StreamlitUI
    } else {
        # Start both services
        $apiJob = Start-ApiServer
        Start-StreamlitUI
    }
} finally {
    # Cleanup
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Yellow
    
    # Stop background jobs
    Get-Job | Stop-Job
    Get-Job | Remove-Job
    
    # Kill any remaining Python processes (be careful with this)
    try {
        Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.MainWindowTitle -like "*uvicorn*" } | Stop-Process -Force
    } catch {
        # Ignore errors
    }
    
    Write-Host "Shutdown complete." -ForegroundColor Green
}

Write-Host "Press Enter to exit..." -ForegroundColor Yellow
Read-Host

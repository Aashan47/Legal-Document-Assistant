@echo off
REM Windows batch script to start the Legal Document Analyzer

echo =====================================================
echo    Legal Document Analyzer - Windows Startup
echo =====================================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt

REM Create .env file if it doesn't exist
if not exist ".env" (
    echo Creating .env file from template...
    copy .env.example .env
    echo.
    echo IMPORTANT: Please edit .env file with your API keys before proceeding
    echo Press any key to continue after editing .env file...
    pause
)

REM Start the application
echo.
echo Starting Legal Document Analyzer...
echo.
echo API will be available at: http://localhost:8000
echo Web UI will be available at: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

REM Start API server in background
start "Legal Doc Analyzer API" cmd /c "python -m uvicorn src.api.main:app --host localhost --port 8000 --reload"

REM Wait a bit for API to start
timeout /t 5 /nobreak >nul

REM Start Streamlit UI
python -m streamlit run src\ui\app.py --server.port 8501

REM Clean up
echo.
echo Shutting down...
taskkill /f /im python.exe /fi "windowtitle eq Legal Doc Analyzer API*" >nul 2>&1

pause

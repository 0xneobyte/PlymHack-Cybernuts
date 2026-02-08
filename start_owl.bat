@echo off
REM ========================================
REM  OWL ENGINE - Startup Script (Windows)
REM ========================================
echo.
echo ========================================
echo    🦉 OWL ENGINE - Starting System
echo ========================================
echo.

cd /d "%~dp0"

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found! Please install Python 3.11 or higher
    echo    Download from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✓ Python found
python --version

REM Change to owl_engine directory
cd owl_engine

REM Check if dependencies are installed
echo.
echo Checking dependencies...
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo.
    echo 📦 Installing dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ Failed to install dependencies
        pause
        exit /b 1
    )
)

echo ✓ Dependencies installed

REM Create data directories
if not exist "owl_data" mkdir owl_data
if not exist "traffic_data_collected" mkdir traffic_data_collected
if not exist "intelligence_events" mkdir intelligence_events

echo.
echo ========================================
echo  Starting OWL Engine Components
echo ========================================
echo.

REM Start data collection in background
echo 🔄 Starting continuous data collection...
start "OWL Data Collector" /MIN python continuous_collector.py
timeout /t 3 /nobreak >nul

echo ✓ Data collector running in background
echo.

REM Start the Palantir dashboard
echo 🎯 Starting Palantir Intelligence Dashboard...
echo.
echo ========================================
echo  Dashboard will open in your browser
echo  URL: http://localhost:8501
echo ========================================
echo.
echo Press Ctrl+C to stop the dashboard
echo (Data collection will continue in background)
echo.

python -m streamlit run palantir_dashboard.py

echo.
echo Dashboard stopped.
pause

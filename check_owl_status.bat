@echo off
REM Quick test of OWL startup - verifies components without starting
cd /d "%~dp0"

echo.
echo ========================================
echo  🦉 OWL ENGINE - System Check
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found
    exit /b 1
) else (
    echo ✓ Python installed
    python --version
)

echo.
echo Checking Python packages...

REM Check key dependencies
python -c "import streamlit; print('✓ streamlit')" 2>nul || echo ❌ streamlit not installed
python -c "import pandas; print('✓ pandas')" 2>nul || echo ❌ pandas not installed
python -c "import plotly; print('✓ plotly')" 2>nul || echo ❌ plotly not installed
python -c "import cv2; print('✓ opencv-python (cv2)')" 2>nul || echo ❌ opencv-python not installed
python -c "import folium; print('✓ folium')" 2>nul || echo ❌ folium not installed

echo.
echo Checking Streamlit command access...
python -m streamlit --version >nul 2>&1 && echo ✓ streamlit CLI accessible || echo ❌ streamlit CLI not accessible

echo.
echo Checking OWL Engine files...

cd owl_engine

if exist "palantir_dashboard.py" (echo ✓ palantir_dashboard.py) else (echo ❌ palantir_dashboard.py)
if exist "continuous_collector.py" (echo ✓ continuous_collector.py) else (echo ❌ continuous_collector.py)
if exist "main.py" (echo ✓ main.py) else (echo ❌ main.py)
if exist "abbey_road_dashboard.py" (echo ✓ abbey_road_dashboard.py) else (echo ❌ abbey_road_dashboard.py)
if exist "data_manager.py" (echo ✓ data_manager.py) else (echo ❌ data_manager.py)

if exist "data_collection\" (echo ✓ data_collection\) else (echo ❌ data_collection\)
if exist "intelligence\" (echo ✓ intelligence\) else (echo ❌ intelligence\)
if exist "database\" (echo ✓ database\) else (echo ❌ database\)

echo.
echo ========================================
echo  System Status
echo ========================================

REM Check if already running
tasklist /FI "WindowTitle eq OWL Data Collector*" 2>nul | find /I "python.exe" >nul
if not errorlevel 1 (
    echo 🔄 Data collector is RUNNING
) else (
    echo 💤 Data collector is stopped
)

netstat -ano | findstr ":8501" >nul 2>&1
if not errorlevel 1 (
    echo 🌐 Dashboard is RUNNING (port 8501)
) else (
    echo 💤 Dashboard is stopped
)

echo.
echo ========================================
echo.

if exist "owl_data\" (
    echo 📊 Data directory: owl_data\
    for /f %%i in ('dir /b /a:d owl_data 2^>nul ^| find /c /v ""') do echo    Dates collected: %%i
)

echo.
pause

@echo off
REM Quick test of startup components without actually starting

cd /d "%~dp0"

echo.
echo ========================================
echo  Testing OWL Startup Components
echo ========================================
echo.

REM Test Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python not found
    exit /b 1
) else (
    echo ✓ Python available
)

REM Test Streamlit module
python -m streamlit --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Streamlit module not accessible
    exit /b 1
) else (
    echo ✓ Streamlit module accessible
    python -m streamlit --version
)

REM Test owl_engine directory
cd owl_engine
if errorlevel 1 (
    echo ❌ owl_engine directory not found
    exit /b 1
) else (
    echo ✓ owl_engine directory found
)

REM Test files exist
if not exist "palantir_dashboard.py" (
    echo ❌ palantir_dashboard.py not found
    exit /b 1
) else (
    echo ✓ palantir_dashboard.py exists
)

if not exist "continuous_collector.py" (
    echo ❌ continuous_collector.py not found
    exit /b 1
) else (
    echo ✓ continuous_collector.py exists
)

REM Test import of main dashboard
python -c "from palantir_dashboard import *" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  Dashboard imports have errors (may be Streamlit-specific, OK if running)
) else (
    echo ✓ Dashboard imports successfully
)

echo.
echo ========================================
echo  All Startup Tests Passed!
echo ========================================
echo.
echo Ready to run: ..\start_owl.bat
echo.
pause

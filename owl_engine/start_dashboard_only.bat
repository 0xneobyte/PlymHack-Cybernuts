@echo off
REM ========================================
REM  Start Palantir Dashboard Only
REM ========================================
cd /d "%~dp0"

echo.
echo 🦉 Starting OWL Palantir Dashboard...
echo.
echo URL: http://localhost:8501
echo.

python -m streamlit run palantir_dashboard.py

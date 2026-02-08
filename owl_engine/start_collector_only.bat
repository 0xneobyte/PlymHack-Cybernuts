@echo off
REM ========================================
REM  Start Data Collector Only
REM ========================================
cd /d "%~dp0"

echo.
echo 🦉 Starting OWL Continuous Data Collector...
echo.
echo This will run in the foreground.
echo Press Ctrl+C to stop.
echo.

python continuous_collector.py

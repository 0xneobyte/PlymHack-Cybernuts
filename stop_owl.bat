@echo off
REM ========================================
REM  OWL ENGINE - Stop Script (Windows)
REM ========================================
echo.
echo ========================================
echo    🦉 OWL ENGINE - Stopping System
echo ========================================
echo.

REM Stop data collector
echo Stopping data collector...
taskkill /FI "WindowTitle eq OWL Data Collector*" /F >nul 2>&1

REM Stop any running Python processes related to OWL
echo Stopping continuous_collector.py...
wmic process where "commandline like '%%continuous_collector.py%%'" delete >nul 2>&1

REM Stop Streamlit dashboards
echo Stopping dashboards...
for /f "tokens=2" %%a in ('tasklist /FI "IMAGENAME eq python.exe" /FO LIST ^| findstr /I "PID"') do (
    wmic process where "ProcessId=%%a and CommandLine like '%%streamlit%%'" delete >nul 2>&1
)

echo.
echo ✓ All OWL Engine components stopped
echo.
pause

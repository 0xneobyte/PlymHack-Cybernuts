#!/bin/bash
# ========================================
#  OWL ENGINE - Startup Script (Linux/Mac)
# ========================================

echo ""
echo "========================================"
echo "   🦉 OWL ENGINE - Starting System"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3.11 or higher"
    exit 1
fi

echo "✓ Python found"
python3 --version

# Change to owl_engine directory
cd owl_engine

# Check if dependencies are installed
echo ""
echo "Checking dependencies..."
if ! python3 -c "import streamlit" &> /dev/null; then
    echo ""
    echo "📦 Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✓ Dependencies installed"

# Create data directories
mkdir -p owl_data
mkdir -p traffic_data_collected
mkdir -p intelligence_events

echo ""
echo "========================================"
echo " Starting OWL Engine Components"
echo "========================================"
echo ""

# Start data collection in background
echo "🔄 Starting continuous data collection..."
python3 continuous_collector.py > owl_collector.log 2>&1 &
COLLECTOR_PID=$!
echo "✓ Data collector running (PID: $COLLECTOR_PID)"
sleep 2

echo ""
echo "🎯 Starting Palantir Intelligence Dashboard..."
echo ""
echo "========================================"
echo " Dashboard will open in your browser"
echo " URL: http://localhost:8501"
echo "========================================"
echo ""
echo "Press Ctrl+C to stop the dashboard"
echo "(Data collection will continue in background)"
echo ""

# Trap Ctrl+C to clean up
trap "echo ''; echo 'Stopping dashboard...'; kill $COLLECTOR_PID 2>/dev/null; exit" INT TERM

# Start Streamlit dashboard
streamlit run palantir_dashboard.py

echo ""
echo "Dashboard stopped."
echo "Data collector still running (PID: $COLLECTOR_PID)"
echo "To stop: kill $COLLECTOR_PID"

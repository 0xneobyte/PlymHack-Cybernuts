#!/bin/bash
# Quick test of OWL startup - verifies components without starting

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "========================================"
echo " 🦉 OWL ENGINE - System Check"
echo "========================================"
echo ""

# Check Python
if command -v python3 &> /dev/null; then
    echo "✓ Python installed"
    python3 --version
else
    echo "❌ Python not found"
    exit 1
fi

echo ""
echo "Checking Python packages..."

# Check key dependencies
python3 -c "import streamlit; print('✓ streamlit')" 2>/dev/null || echo "❌ streamlit not installed"
python3 -c "import pandas; print('✓ pandas')" 2>/dev/null || echo "❌ pandas not installed"
python3 -c "import plotly; print('✓ plotly')" 2>/dev/null || echo "❌ plotly not installed"
python3 -c "import cv2; print('✓ opencv-python (cv2)')" 2>/dev/null || echo "❌ opencv-python not installed"
python3 -c "import folium; print('✓ folium')" 2>/dev/null || echo "❌ folium not installed"

echo ""
echo "Checking OWL Engine files..."

cd owl_engine

[ -f "palantir_dashboard.py" ] && echo "✓ palantir_dashboard.py" || echo "❌ palantir_dashboard.py"
[ -f "continuous_collector.py" ] && echo "✓ continuous_collector.py" || echo "❌ continuous_collector.py"
[ -f "main.py" ] && echo "✓ main.py" || echo "❌ main.py"
[ -f "abbey_road_dashboard.py" ] && echo "✓ abbey_road_dashboard.py" || echo "❌ abbey_road_dashboard.py"
[ -f "data_manager.py" ] && echo "✓ data_manager.py" || echo "❌ data_manager.py"

[ -d "data_collection" ] && echo "✓ data_collection/" || echo "❌ data_collection/"
[ -d "intelligence" ] && echo "✓ intelligence/" || echo "❌ intelligence/"
[ -d "database" ] && echo "✓ database/" || echo "❌ database/"

echo ""
echo "========================================"
echo " System Status"
echo "========================================"

# Check if already running
if pgrep -f "continuous_collector.py" > /dev/null; then
    echo "🔄 Data collector is RUNNING (PID: $(pgrep -f continuous_collector.py))"
else
    echo "💤 Data collector is stopped"
fi

if lsof -i :8501 > /dev/null 2>&1; then
    echo "🌐 Dashboard is RUNNING (port 8501)"
else
    echo "💤 Dashboard is stopped"
fi

echo ""
echo "========================================"
echo ""

if [ -d "owl_data" ]; then
    echo "📊 Data directory: owl_data/"
    DATES=$(ls -1 owl_data 2>/dev/null | wc -l)
    echo "   Dates collected: $DATES"
fi

echo ""

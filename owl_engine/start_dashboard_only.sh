#!/bin/bash
# ========================================
#  Start Palantir Dashboard Only
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🦉 Starting OWL Palantir Dashboard..."
echo ""
echo "URL: http://localhost:8501"
echo ""

streamlit run palantir_dashboard.py

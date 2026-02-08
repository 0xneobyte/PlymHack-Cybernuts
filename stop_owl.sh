#!/bin/bash
# ========================================
#  OWL ENGINE - Stop Script (Linux/Mac)
# ========================================

echo ""
echo "========================================"
echo "   🦉 OWL ENGINE - Stopping System"
echo "========================================"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Stopping data collector..."
pkill -f "continuous_collector.py"

echo "Stopping dashboards..."
pkill -f "streamlit.*palantir_dashboard"
pkill -f "streamlit"

echo ""
echo "✓ All OWL Engine components stopped"
echo ""

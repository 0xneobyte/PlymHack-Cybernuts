#!/bin/bash
# ========================================
#  Start Data Collector Only
# ========================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "🦉 Starting OWL Continuous Data Collector..."
echo ""
echo "This will run in the foreground."
echo "Press Ctrl+C to stop."
echo ""

python3 continuous_collector.py

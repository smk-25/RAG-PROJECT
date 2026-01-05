#!/bin/bash

# Startup script for Summarizationcode.py
# This script ensures proper cache clearing and file watching

echo "Starting Summarizationcode.py with Streamlit..."

# Clear any Python cache
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# Run streamlit with proper configuration
streamlit run Summarizationcode.py \
    --server.fileWatcherType=auto \
    --runner.fastReruns=true \
    --server.address=0.0.0.0 \
    --server.port=8501

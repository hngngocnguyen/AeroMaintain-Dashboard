#!/bin/bash

# AeroMaintain Dashboard - Setup and Launch Script

echo ""
echo "============================================================"
echo "  🛩️  AeroMaintain Dashboard - Streamlit Application"
echo "============================================================"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: Python3 is not installed or not in PATH"
    exit 1
fi

echo "✅ Python found:"
python3 --version

# Install dependencies
echo ""
echo "📦 Installing dependencies from streamlit_requirements.txt..."
echo ""

pip install -r streamlit_requirements.txt

if [ $? -ne 0 ]; then
    echo "❌ ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "✅ Dependencies installed successfully!"
echo ""

# Launch Streamlit
echo "============================================================"
echo "  🚀 Launching Streamlit Application..."
echo "============================================================"
echo ""
echo "📊 The dashboard will open at: http://localhost:8501"
echo "📌 To stop the server, press Ctrl+C"
echo ""

streamlit run app.py

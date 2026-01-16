@echo off
REM AeroMaintain Dashboard - Setup and Launch Script

echo.
echo ============================================================
echo  🛩️  AeroMaintain Dashboard - Streamlit Application
echo ============================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo ✅ Python found: 
python --version

REM Install dependencies
echo.
echo 📦 Installing dependencies from streamlit_requirements.txt...
echo.

pip install -r streamlit_requirements.txt

if %errorlevel% neq 0 (
    echo ❌ ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo ✅ Dependencies installed successfully!
echo.

REM Launch Streamlit
echo ============================================================
echo  🚀 Launching Streamlit Application...
echo ============================================================
echo.
echo 📊 The dashboard will open at: http://localhost:8501
echo 📌 To stop the server, press Ctrl+C
echo.

streamlit run app.py

pause

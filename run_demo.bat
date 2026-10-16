@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Project setup is not complete. Double-click setup_windows.bat once before the presentation.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -c "import streamlit, plotly, pandas, sklearn, yaml" >nul 2>nul
if errorlevel 1 (
    echo Some project packages are missing. Double-click setup_windows.bat to repair the setup.
    pause
    exit /b 1
)

echo Starting the Sensor Trust dashboard. Keep this window open while presenting.
echo Press Ctrl+C here to stop the dashboard.
".venv\Scripts\python.exe" -m streamlit run app/dashboard.py

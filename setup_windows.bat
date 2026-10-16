@echo off
setlocal
cd /d "%~dp0"

where py >nul 2>nul
if errorlevel 1 (
    where python >nul 2>nul
    if errorlevel 1 (
        echo Python 3.10 or newer was not found. Install Python from https://www.python.org/downloads/ and select Add Python to PATH.
        pause
        exit /b 1
    )
    set "PYTHON=python"
) else (
    set "PYTHON=py -3"
)

echo Creating the project environment...
%PYTHON% -m venv .venv
if errorlevel 1 goto failed

echo Installing project packages. This needs an internet connection and may take a few minutes...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed

echo.
echo Setup complete. Double-click run_demo.bat to start the presentation dashboard.
pause
exit /b 0

:failed
echo.
echo Setup did not finish. Check the error above, then run setup_windows.bat again.
pause
exit /b 1

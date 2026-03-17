@echo off
echo ============================================
echo   Fake News Detector - With Public URL
echo ============================================
echo.

REM Check if Node.js is installed (needed for localtunnel)
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Node.js is not installed.
    echo Please install Node.js from https://nodejs.org
    echo Or use ngrok from https://ngrok.com
    echo.
    echo For now, starting local server only...
    echo You can access at: http://localhost:5000
    echo.
    cd /d "%~dp0"
    python app.py
    goto :end
)

echo Starting Flask server and creating public URL...
echo.
cd /d "%~dp0"

REM Start Flask in background
start "Flask Server" python app.py

REM Wait for server to start
timeout /t 3 /nobreak >nul

REM Create public URL using localtunnel
echo.
echo ============================================
echo Opening public URL...
echo If prompted for a port, enter: 5000
echo ============================================
node -e "require('child_process').exec('npx localtunnel --port 5000')"

:end
pause


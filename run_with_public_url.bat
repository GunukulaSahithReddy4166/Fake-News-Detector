@echo off
title Fake News Detector - Starting...
echo ============================================
echo   Fake News Detector with Public URL
echo ============================================
echo.

cd /d "%~dp0"

REM Check if ngrok exists
if not exist "ngrok.exe" (
    echo ERROR: ngrok.exe not found!
    echo Please download ngrok from https://ngrok.com
    pause
    exit /b 1
)

echo Starting Flask server on port 5000...
start "Flask" python app.py

REM Wait for Flask to start
timeout /t 5 /nobreak >nul

echo.
echo Creating public URL with ngrok...
echo.
echo NOTE: If this is your first time using ngrok, you may need to:
echo 1. Sign up at https://ngrok.com
echo 2. Copy your auth token: ngrok authtoken YOUR_TOKEN
echo.

ngrok http 5000

pause


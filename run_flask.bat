@echo off
title Fake News Detector
echo.
echo ========================================
echo   FAKE NEWS DETECTOR STARTUP
echo ========================================
echo.

echo [1/4] Installing dependencies...
call pip install -r requirements.txt

echo.
echo [2/4] Training ML model...
call python train_model.py

echo.
echo [3/4] Starting Flask server...
call python app.py

echo.
echo [4/4] Opening browser...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:5000

echo.
echo ========================================
echo   ✅ WEBSITE READY AT: http://127.0.0.1:5000
echo   📱 Press Ctrl+C to stop server
echo ========================================
pause

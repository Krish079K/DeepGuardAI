@echo off
title DeepGuard AI - Public Live Online Server
echo =================================================================
echo   DeepGuard AI - Launch Live Server with Public HTTPS Tunnel
echo =================================================================
echo.
echo Starting local backend server on 0.0.0.0:8000...
start "DeepGuard_Backend" cmd /k "cd /d %~dp0 && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

echo Waiting for backend to initialize...
timeout /t 3 >nul

echo.
echo Generating public HTTPS tunnel for mobile app & remote web access...
echo (You can enter this generated URL into the DeepGuard Mobile App!)
echo.
npx -y localtunnel --port 8000
pause

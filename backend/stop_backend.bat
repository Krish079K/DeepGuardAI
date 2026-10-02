@echo off
title Stop DeepGuard AI Backend
echo Stopping any running DeepGuard AI backend processes...
taskkill /F /IM python.exe /FI "WINDOWTITLE eq DeepGuard*" 2>nul
for /f "tokens=5" %%a in ('netstat -aon ^| find ":8000" ^| find "LISTENING"') do (
    echo Terminating PID %%a on port 8000...
    taskkill /F /PID %%a 2>nul
)
echo [DONE] DeepGuard AI backend stopped.
timeout /t 3 >nul

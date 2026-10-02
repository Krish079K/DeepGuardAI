@echo off
title DeepGuard AI - Auto-Start Setup
echo =======================================================
echo   DeepGuard AI - Setup Permanent Auto-Start on Windows
echo =======================================================
echo.
echo Setting up DeepGuard AI backend to start automatically
echo in the background whenever Windows boots up...
echo.

set SCRIPT_DIR=%~dp0
set VBS_TARGET=%SCRIPT_DIR%start_backend_silent.vbs
set STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_PATH=%STARTUP_FOLDER%\DeepGuard_Backend.vbs

copy /Y "%VBS_TARGET%" "%SHORTCUT_PATH%" >nul

if %errorlevel% equ 0 (
    echo [SUCCESS] DeepGuard AI backend will now automatically run
    echo           in the background every time your computer starts!
    echo.
    echo Shortcut created at:
    echo %SHORTCUT_PATH%
    echo.
    echo Starting the backend now...
    wscript "%VBS_TARGET%"
    echo [OK] Backend is now running live at http://localhost:8000
) else (
    echo [ERROR] Could not install to Startup folder.
)

echo.
pause

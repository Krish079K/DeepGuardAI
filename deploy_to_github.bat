@echo off
title Push DeepGuard AI to GitHub
echo ========================================================
echo   DeepGuard AI - One-Click GitHub Setup for Cloud Host
echo ========================================================
echo.
echo Make sure you have created an empty repository on https://github.com/new
echo.
set /p REPO_URL="Enter your GitHub Repository URL (e.g. https://github.com/username/DeepGuardAI.git): "

if "%REPO_URL%"=="" (
    echo [ERROR] No URL entered.
    pause
    exit /b
)

echo.
echo Adding remote origin...
git branch -M main
git remote remove origin 2>nul
git remote add origin %REPO_URL%

echo.
echo Pushing code to GitHub...
git push -u origin main

if %errorlevel% equ 0 (
    echo.
    echo ========================================================
    echo [SUCCESS] Code uploaded to GitHub!
    echo.
    echo Now you can deploy 24/7 on Render.com:
    echo 1. Go to https://dashboard.render.com
    echo 2. Click "New +" -> "Web Service"
    echo 3. Connect your GitHub repository
    echo 4. Set Root Directory to: backend
    echo 5. Click "Create Web Service" (Free Plan)
    echo ========================================================
) else (
    echo.
    echo [NOTE] If prompted for GitHub login, please sign in or use a GitHub Personal Access Token.
)

echo.
pause

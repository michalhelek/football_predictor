@echo off
chcp 65001 >nul
cd /d "%~dp0.."
echo.
echo Football Predictor - instalacja
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1"
if errorlevel 1 (
    echo.
    pause
    exit /b 1
)
echo.
pause

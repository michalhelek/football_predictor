@echo off
chcp 65001 >nul
cd /d "%~dp0.."

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo Brak srodowiska .venv — uruchom najpierw installer\install.bat
    pause
    exit /b 1
)

echo Uruchamianie Football Predictor...
echo Aplikacja otworzy sie w przegladarce: http://localhost:8501
echo Zamknij to okno, aby zatrzymac serwer.
echo.

start "" "http://localhost:8501"
"%VENV_PY%" -m streamlit run app.py --server.headless true

pause

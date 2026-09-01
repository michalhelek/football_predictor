# Football Predictor — instalacja zależności (Windows)
# Uruchom: kliknij install.bat lub: powershell -ExecutionPolicy Bypass -File installer\install.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

Write-Host ""
Write-Host "=== Football Predictor — instalacja ===" -ForegroundColor Cyan
Write-Host "Folder: $Root"
Write-Host ""

function Test-Python {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) {
        Write-Host "BLAD: Nie znaleziono Pythona." -ForegroundColor Red
        Write-Host "Pobierz Python 3.10+ z https://www.python.org/downloads/"
        Write-Host "Przy instalacji zaznacz: Add python.exe to PATH"
        exit 1
    }
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "BLAD: Wymagany Python 3.10 lub nowszy." -ForegroundColor Red
        exit 1
    }
    $ver = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
    Write-Host "Python: $ver"
}

Test-Python

$venvPython = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Host "Tworzenie wirtualnego srodowiska (.venv)..."
    python -m venv .venv
}

Write-Host "Instalowanie bibliotek (moze potrwac kilka minut)..."
& $venvPython -m pip install --upgrade pip --quiet
& $venvPython -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "BLAD: Instalacja bibliotek nie powiodla sie." -ForegroundColor Red
    exit 1
}

$dataDir = Join-Path $Root "data"
$outputDir = Join-Path $Root "output"
New-Item -ItemType Directory -Force -Path $dataDir, $outputDir | Out-Null

$tokenPath = Join-Path $dataDir ".api_token"
$tokenExample = Join-Path $dataDir ".api_token.example"
if (-not (Test-Path $tokenPath)) {
    if (Test-Path $tokenExample) {
        Copy-Item $tokenExample $tokenPath
        Write-Host "Utworzono data\.api_token — wklej token z https://www.football-data.org/client/register"
    } else {
        Write-Host "UWAGA: Brak pliku data\.api_token — terminarz API moze nie dzialac."
    }
}

Write-Host ""
Write-Host "Instalacja zakonczona pomyslnie." -ForegroundColor Green
Write-Host ""
Write-Host "Nastepne kroki:"
Write-Host "  1. Edytuj data\.api_token (token API football-data.org)"
Write-Host "  2. Kliknij: installer\Uruchom aplikacje.bat"
Write-Host "     lub w terminalu: .venv\Scripts\python.exe main.py init"
Write-Host ""

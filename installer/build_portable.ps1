# Buduje paczke ZIP z wbudowanym Pythonem (bez instalacji Pythona na docelowym PC).
# Uruchom: powershell -ExecutionPolicy Bypass -File installer\build_portable.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$DistName = "FootballPredictor_Portable"
$DistRoot = Join-Path $env:TEMP $DistName
$ZipPath = Join-Path $Root "$DistName.zip"
$PythonVersion = "3.12.7"
$EmbedZip = "python-$PythonVersion-embed-amd64.zip"
$EmbedUrl = "https://www.python.org/ftp/python/$PythonVersion/$EmbedZip"

Write-Host "=== Budowanie paczki przenosnej ===" -ForegroundColor Cyan
Write-Host "Docelowy plik: $ZipPath"

if (Test-Path $DistRoot) {
    Remove-Item $DistRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $DistRoot | Out-Null

$skipNames = @(".git", ".venv", "__pycache__", ".ipynb_checkpoints", "installer")
Get-ChildItem $Root -Force | ForEach-Object {
    if ($_.Name -in $skipNames) { return }
    $dest = Join-Path $DistRoot $_.Name
    if ($_.PSIsContainer) {
        Copy-Item $_.FullName $dest -Recurse -Force
    } else {
        Copy-Item $_.FullName $dest -Force
    }
}

New-Item -ItemType Directory -Force -Path (Join-Path $DistRoot "installer") | Out-Null
Copy-Item (Join-Path $PSScriptRoot "install.ps1") (Join-Path $DistRoot "installer\install.ps1") -Force
Copy-Item (Join-Path $PSScriptRoot "install.bat") (Join-Path $DistRoot "installer\install.bat") -Force
Copy-Item (Join-Path $PSScriptRoot "Uruchom aplikacje.bat") (Join-Path $DistRoot "installer\Uruchom aplikacje.bat") -Force

$token = Join-Path $DistRoot "data\.api_token"
if (Test-Path $token) { Remove-Item $token -Force }
Get-ChildItem (Join-Path $DistRoot "data") -Filter "*.joblib" -ErrorAction SilentlyContinue | Remove-Item -Force
$db = Join-Path $DistRoot "data\matches.db"
if (Test-Path $db) { Remove-Item $db -Force }
if (Test-Path (Join-Path $DistRoot "data\.api_token.example")) {
    Copy-Item (Join-Path $DistRoot "data\.api_token.example") (Join-Path $DistRoot "data\.api_token")
}

Write-Host "Pobieranie Pythona $PythonVersion (embeddable)..."
$tempZip = Join-Path $env:TEMP $EmbedZip
Invoke-WebRequest -Uri $EmbedUrl -OutFile $tempZip -UseBasicParsing
$pythonDir = Join-Path $DistRoot "python"
Expand-Archive -Path $tempZip -DestinationPath $pythonDir -Force
Remove-Item $tempZip -Force

$sitePackages = Join-Path $pythonDir "Lib\site-packages"
New-Item -ItemType Directory -Force -Path $sitePackages | Out-Null

$pthFile = Get-ChildItem $pythonDir -Filter "python*._pth" | Select-Object -First 1
$pthContent = @(
    "python312.zip",
    ".",
    "Lib\site-packages",
    "",
    "import site"
)
Set-Content -Path $pthFile.FullName -Value $pthContent -Encoding ASCII

$pyExe = Join-Path $pythonDir "python.exe"
Write-Host "Instalowanie pip i bibliotek (kilka minut)..."
$getPip = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $getPip -UseBasicParsing
& $pyExe $getPip --no-warn-script-location
& $pyExe -m pip install -r (Join-Path $DistRoot "requirements.txt") --no-warn-script-location
if ($LASTEXITCODE -ne 0) {
    Write-Host "BLAD: Instalacja bibliotek nie powiodla sie." -ForegroundColor Red
    exit 1
}

@'
@echo off
chcp 65001 >nul
cd /d "%~dp0"

if not exist "python\python.exe" (
    echo Brak folderu python — pobierz pelna paczke ponownie.
    pause
    exit /b 1
)

echo Football Predictor
echo Otwieram http://localhost:8501
echo Zamknij to okno, aby zatrzymac aplikacje.
echo.

start "" "http://localhost:8501"
"python\python.exe" -m streamlit run app.py --server.headless true
pause
'@ | Set-Content -Path (Join-Path $DistRoot "URUCHOM.bat") -Encoding UTF8

Copy-Item (Join-Path $PSScriptRoot "INSTRUKCJA_INSTALACJI.txt") (Join-Path $DistRoot "INSTRUKCJA_INSTALACJI.txt") -Force

if (Test-Path $ZipPath) { Remove-Item $ZipPath -Force }
Compress-Archive -Path $DistRoot -DestinationPath $ZipPath -Force

$sizeMb = [math]::Round((Get-Item $ZipPath).Length / 1MB, 1)
Write-Host ""
Write-Host "Gotowe: $ZipPath ($sizeMb MB)" -ForegroundColor Green
Write-Host "Skopiuj ZIP na inny komputer (pendrive, chmura), rozpakuj i uruchom URUCHOM.bat"

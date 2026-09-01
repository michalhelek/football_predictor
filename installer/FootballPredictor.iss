; Inno Setup — opcjonalny instalator .exe
; Pobierz Inno Setup: https://jrsoftware.org/isinfo.php
; Skompiluj ten plik, aby utworzyc FootballPredictor_Setup.exe

#define MyAppName "Football Predictor"
#define MyAppVersion "1.0"
#define MyAppPublisher "Football Predictor"
#define MyAppURL "https://github.com"
#define MyAppExeName "Uruchom aplikacje.bat"
#define SourceDir ".."

[Setup]
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputBaseFilename=FootballPredictor_Setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "polish"; MessagesFile: "compiler:Languages\Polish.isl"

[Tasks]
Name: "desktopicon"; Description: "Utworz skrot na pulpicie"; GroupDescription: "Skroty:"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs; Excludes: ".git\*,.venv\*,__pycache__\*,.ipynb_checkpoints\*,\.api_token,installer\dist\*"
Source: "{#SourceDir}\data\.api_token.example"; DestDir: "{app}\data"; DestName: ".api_token"; Flags: ignoreversion onlyifdoesntexist

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\installer\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\installer\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "powershell.exe"; Parameters: "-NoProfile -ExecutionPolicy Bypass -File ""{app}\installer\install.ps1"""; StatusMsg: "Instalowanie bibliotek Python..."; Flags: waituntilterminated
Filename: "{app}\installer\{#MyAppExeName}"; Description: "Uruchom {#MyAppName}"; Flags: nowait postinstall skipifsilent

[Code]
function InitializeSetup(): Boolean;
begin
  Result := True;
  if not RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.13\InstallPath', '', '') and
     not RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.12\InstallPath', '', '') and
     not RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.11\InstallPath', '', '') and
     not RegQueryStringValue(HKLM, 'SOFTWARE\Python\PythonCore\3.10\InstallPath', '', '') then
  begin
    if MsgBox('Nie wykryto Pythona 3.10+. Instalator skopiuje pliki, ale Python musi byc zainstalowany osobno z python.org (zaznacz Add to PATH). Kontynuowac?', mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end;
end;

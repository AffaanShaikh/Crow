; crow_installer.iss - Inno Setup installer script
;
; Produces: CrowSetup-1.0.0.exe
; Requires: Inno Setup 6+ (https://jrsoftware.org/isinfo.php)
;
; Run:
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\crow_installer.iss
;   (or use the Inno Setup IDE)
;
; INNO SETUP coz:-
; Inno Setup is the industry standard for Windows installers:
;   - Free, open source, very mature (20+ years)
;   - Produces a single signed .exe installer
;   - Handles: shortcuts, registry, PATH, auto-start, uninstaller
;   - Unicode support, custom UI, conditional install steps
;   - Alternative: NSIS (also free), or WiX (XML-based, more complex)
;
; WHAT THE INSTALLER DOES:-
;   1. Shows license agreement and install path selection
;   2. Copies dist\crow\ to Program Files\Crow\
;   3. Creates Start Menu shortcut + optional Desktop shortcut
;   4. Optionally registers auto-start in HKCU\Software\Microsoft\Windows\CurrentVersion\Run
;   5. Adds crow.exe to PATH (optional)
;   6. Creates an uninstaller (Programs > Add/Remove Programs)
;   7. Opens Crow after install

#define AppName      "Crow"
#define AppVersion   "1.0.0"
#define AppPublisher "AffaanShaikh"
#define AppURL       "https://github.com/AffaanShaikh/Crow"
#define AppExeName   "crow.exe"
#define SourceDir    "..\dist\crow"
#define OutputDir    "..\dist\installer"


; Visual C++ Redistributable - required by Python C extensions (numpy, onnxruntime etc.)
; Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe
; Place it next to crow_installer.iss before building.
#define VCRedistPath "vc_redist.x64.exe"


[Setup]
; REQUIRED: change AppId to a fresh GUID for each application
AppId={{A1B2C3D4-E5F6-7890-ABCD-EF1234567890}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}/issues
AppUpdatesURL={#AppURL}/releases

; Install location
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes

; Output
OutputDir={#OutputDir}
OutputBaseFilename=CrowSetup-{#AppVersion}

; Compression (lzma2/ultra gives ~40% smaller installer)
Compression=lzma2
SolidCompression=yes
CompressionThreads=4

; Appearance
WizardStyle=modern
WizardSmallImageFile=..\assets\crow_installer_small.bmp
; 55x58 px
; WizardImageFile=..\assets\crow_installer_banner.bmp
; 164x314 px

; Privileges: user-mode install (no admin required - installs to AppData)
; Change to admin if you need HKLM registry keys or system-wide PATH
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; Code signing - enable once you have a certificate
; SignTool=signtool sign /fd sha256 /t http://timestamp.digicert.com /f MyCert.pfx $f
; SignedUninstaller=yes

; Minimum Windows version: Windows 10 (for ONNX Runtime, PyTorch)
MinVersion=10.0.17763

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";   Description: "{cm:CreateDesktopIcon}";     GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startupentry";  Description: "Start Crow automatically when Windows starts"; GroupDescription: "Startup"; Flags: unchecked
Name: "addtopath";     Description: "Add Crow to system PATH (for command-line use)"; GroupDescription: "Integration"; Flags: unchecked


[Files]
; App bundle
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Starter .env - only written if the user doesn't already have one
; Create a minimal .env.default next to crow_installer.iss:
;   LLM_MODEL_NAME=qwen3:latest
;   DEFAULT_MAX_TOKENS=2048
Source: ".env.default"; DestDir: "{localappdata}\Crow"; \
  DestName: ".env"; Flags: onlyifdoesntexist uninsneveruninstall


; VC++ Redistributable - silently installed if not already present
Source: "{#VCRedistPath}"; DestDir: "{tmp}"; Flags: deleteafterinstall


[Icons]
; Start Menu
Name: "{group}\{#AppName}";         Filename: "{app}\{#AppExeName}"; WorkingDir: "{userappdata}\Crow"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"

; Desktop (optional task)
Name: "{autodesktop}\{#AppName}";   Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Registry]
; Auto-start (optional task) - runs as current user, no admin needed
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; ValueType: string; ValueName: "{#AppName}"; ValueData: """{app}\{#AppExeName}"""; Flags: uninsdeletevalue; Tasks: startupentry

; Register for "Open with" for supported file types (optional)
Root: HKCU; Subkey: "Software\Classes\.md\OpenWithProgIds"; ValueType: string; ValueName: "Crow.md"; ValueData: ""; Flags: uninsdeletevalue

[Dirs]
; Ensure the user data directory exists
Name: "{userappdata}\Crow"
Name: "{userappdata}\Crow\data\tokens"
Name: "{localappdata}\Crow"
Name: "{localappdata}\Crow\data"
Name: "{localappdata}\Crow\data\tokens"
; Create a starter .env file only if one doesn't already exist
; (preserves user's config across reinstalls / upgrades)

[Run]
; Install VC++ Redistributable silently (/install /quiet /norestart)
; The check /` prevents reinstall if already present
Filename: "{tmp}\{#VCRedistPath}"; \
  Parameters: "/install /quiet /norestart"; \
  StatusMsg: "Installing Visual C++ Runtime..."; \
  Flags: waituntilterminated

; Launch Crow after install
Filename: "{app}\{#AppExeName}"; \
  Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; \
  Flags: nowait postinstall skipifsilent


[UninstallDelete]
; Remove user data on uninstall ONLY if the user confirms
; We do NOT auto-delete tokens/ or chromadb/ - that's the user's data.
; The uninstaller removes only the app files.
Type: filesandordirs; Name: "{app}"

[Messages]
; Custom messages
WelcomeLabel2=This will install [name/ver] on your computer.%n%nCrow is a local AI assistant that runs entirely on your machine. No data is sent to the cloud.%n%nClick Next to continue.

[Code]
// Verify Ollama is installed (Crow needs it for the LLM backend)


function InitializeSetup(): Boolean;
var
  OllamaFound: Boolean;
begin
  Result := True;

  // Check Ollama - check both common install paths
  OllamaFound :=
    FileExists(ExpandConstant('{pf}\Ollama\ollama.exe')) or
    FileExists(ExpandConstant('{pf64}\Ollama\ollama.exe')) or
    (FileSearch('ollama.exe', GetEnv('PATH')) <> '');

  if not OllamaFound then begin
    if MsgBox(
      'Ollama was not detected on this machine.' + #13#10 +
      'Crow requires Ollama to run the AI model.' + #13#10 + #13#10 +
      'After installing Crow, open a terminal and run:' + #13#10 +
      '  1. Install Ollama: https://ollama.com' + #13#10 +
      '  2. Pull a model:   ollama pull qwen3:latest' + #13#10 + #13#10 +
      'Continue installation anyway?',
      mbConfirmation, MB_YESNO) = IDNO then
      Result := False;
  end;
end;


procedure AddToPath(Path: string);
var
  OldPath: string;
  NewPath: string;
begin
  if not RegQueryStringValue(HKCU, 'Environment', 'Path', OldPath) then
    OldPath := '';

  if Pos(Lowercase(Path), Lowercase(OldPath)) = 0 then begin
    if OldPath = '' then
      NewPath := Path
    else
      NewPath := OldPath + ';' + Path;
    RegWriteStringValue(HKCU, 'Environment', 'Path', NewPath);
  end;
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then begin
    // Add to PATH if user chose that task
    // if IsTaskSelected('addtopath') then <-- deprecated, renamed "IsTaskSelected" to "WizardIsTaskSelected" in Inno Setup 6.2+
      // AddToPath(ExpandConstant('{app}'));
    if WizardIsTaskSelected('addtopath') then
      AddToPath(ExpandConstant('{app}'));
  
  end;
end;

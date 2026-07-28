; Inno Setup script for HotS Fever Draft Coach.
; Per-user install (no admin / no UAC): installs to the user's Programs folder,
; adds Start-menu + optional desktop shortcut, and a clean uninstaller.
;
; Build:  ISCC.exe /O"<dist dir>" native\installer\HotSFeverDraftCoach.iss
; (Source path below is relative to this .iss file — build the Release win-x64
;  output first: dotnet build -c Release -r win-x64.)

#define AppName "HotS Fever Draft Coach"
#define AppVersion "0.1.0"
#define AppExe "HotS Fever Draft Coach.exe"
#define SrcDir "..\HotsFever.Overlay\bin\Release\net8.0-windows10.0.19041.0\win-x64"

[Setup]
AppId={{A3F1C2E4-7B9D-4E6A-9C1F-2D8B5E0A7F31}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher=HotS Fever
AppPublisherURL=https://hotsfever.com/draft-coach
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto
PrivilegesRequired=lowest
OutputBaseFilename=HotSFeverDraftCoach-Setup
Compression=lzma2/max
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
WizardStyle=modern
UninstallDisplayIcon={app}\{#AppExe}

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop shortcut"; GroupDescription: "Additional icons:"; Flags: unchecked

[Files]
Source: "{#SrcDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExe}"; Description: "Launch {#AppName}"; Flags: nowait postinstall skipifsilent

; Inno Setup script for the Windows installer (RAFTcorr-Windows-Setup.exe).
;
; Built by `python packaging/build_windows.py`, which stages a private Python, the RAFTcorr wheel
; and the files below, then calls ISCC with /DAppVersion, /DStageDir and /DOutputDir.
;
; Small on purpose: PyTorch is not inside. Right after copying, Setup runs setup_runtime.py with
; the private Python; it installs the PyTorch build that matches the NVIDIA driver (CPU build
; without one), RAFTcorr and its dependencies, and the model files. That needs an internet
; connection once; "Repair RAFTcorr" in the Start menu runs the same step again.
;
; Per-user, as pyALDIC's: no administrator rights. With PrivilegesRequired=lowest, {autopf} is
; %LOCALAPPDATA%\Programs and HKA is HKEY_CURRENT_USER. The output name carries no version, so
; releases/latest/download/RAFTcorr-Windows-Setup.exe keeps working.

#ifndef AppVersion
  #error Pass /DAppVersion=<version>; packaging/build_windows.py does.
#endif
#ifndef NumericVersion
  #error Pass /DNumericVersion=<x.y.z>; packaging/build_windows.py does.
#endif
#ifndef StageDir
  #error Pass /DStageDir=<staged files>; packaging/build_windows.py does.
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist-installers"
#endif

[Setup]
; Never change AppId: it is how an upgrade finds the installation it replaces.
AppId={{31BA150D-6488-4A7D-94FC-C9008281B4FF}
AppName=RAFTcorr
AppVersion={#AppVersion}
AppVerName=RAFTcorr {#AppVersion}
AppPublisher=Zixiang (Zach) Tong, The University of Texas at Austin
AppPublisherURL=https://github.com/zachtong/RAFTcorr
AppSupportURL=https://github.com/zachtong/RAFTcorr/issues
AppUpdatesURL=https://github.com/zachtong/RAFTcorr/releases/latest
VersionInfoVersion={#NumericVersion}
DefaultDirName={autopf}\RAFTcorr
DefaultGroupName=RAFTcorr
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=RAFTcorr-Windows-Setup
SetupIconFile={#StageDir}\RAFTcorr.ico
UninstallDisplayIcon={app}\RAFTcorr.ico
UninstallDisplayName=RAFTcorr
LicenseFile={#StageDir}\LICENSE.txt
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ChangesAssociations=yes
ChangesEnvironment=yes
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "addtopath"; Description: "Add the ""raftcorr"" command to PATH (new terminals)"; GroupDescription: "Command line:"
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[InstallDelete]
; An upgrade starts from a clean private Python: packages of the previous version must not linger.
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\wheels"

[Files]
Source: "{#StageDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\RAFTcorr"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m raftcorr gui"; WorkingDir: "{userdocs}"; IconFilename: "{app}\RAFTcorr.ico"
Name: "{autoprograms}\Repair RAFTcorr"; Filename: "{app}\python\python.exe"; Parameters: """{app}\setup_runtime.py"" --app-dir ""{app}"""; WorkingDir: "{app}"; IconFilename: "{app}\RAFTcorr.ico"
Name: "{autodesktop}\RAFTcorr"; Filename: "{app}\python\pythonw.exe"; Parameters: "-m raftcorr gui"; WorkingDir: "{userdocs}"; IconFilename: "{app}\RAFTcorr.ico"; Tasks: desktopicon

[Registry]
; The keys raftcorr.gui.file_association writes, with the command it uses for an installed Python,
; so the application sees the association as its own and the uninstaller takes it away again.
Root: HKA; Subkey: "Software\Classes\.raftcorr"; ValueType: string; ValueName: ""; ValueData: "RAFTcorr.Project"; Flags: uninsdeletevalue uninsdeletekeyifempty
Root: HKA; Subkey: "Software\Classes\RAFTcorr.Project"; ValueType: string; ValueName: ""; ValueData: "RAFTcorr Project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\RAFTcorr.Project\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\RAFTcorr.ico"
Root: HKA; Subkey: "Software\Classes\RAFTcorr.Project\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\python\pythonw.exe"" -m raftcorr gui ""%1"""
; Only {app}\bin (raftcorr.cmd) goes on PATH, never the private Python or its pip.
Root: HKCU; Subkey: "Environment"; ValueType: expandsz; ValueName: "Path"; ValueData: "{olddata};{app}\bin"; Check: NeedsAddPath('{app}\bin'); Tasks: addtopath

[UninstallDelete]
; Created by setup_runtime.py after the copy, so Inno does not know them.
Type: filesandordirs; Name: "{app}\python"
Type: files; Name: "{app}\install.log"
Type: files; Name: "{app}\runtime.ok"

[Run]
Filename: "{app}\python\pythonw.exe"; Parameters: "-m raftcorr gui"; Description: "{cm:LaunchProgram,RAFTcorr}"; Flags: nowait postinstall skipifsilent; Check: RuntimeReady

[Code]
const
  EnvKey = 'Environment';

function NeedsAddPath(Dir: string): Boolean;
var
  Paths: string;
begin
  Dir := ExpandConstant(Dir);
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths) then
  begin
    Result := True;
    exit;
  end;
  Result := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';') = 0;
end;

procedure RemovePath(Dir: string);
var
  Paths: string;
  P: Integer;
begin
  if not RegQueryStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths) then
    exit;
  P := Pos(';' + Uppercase(Dir) + ';', ';' + Uppercase(Paths) + ';');
  if P = 0 then
    exit;
  if P = 1 then
    Delete(Paths, 1, Length(Dir) + 1)
  else
    Delete(Paths, P - 1, Length(Dir) + 1);
  RegWriteExpandStringValue(HKEY_CURRENT_USER, EnvKey, 'Path', Paths);
end;

function RuntimeReady: Boolean;
begin
  Result := FileExists(ExpandConstant('{app}\runtime.ok'));
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  ResultCode: Integer;
  Params: string;
begin
  if CurStep <> ssPostInstall then
    exit;
  WizardForm.StatusLabel.Caption :=
    'Installing PyTorch, RAFTcorr and the model files (up to 3 GB with a GPU; this takes a while)...';
  Params := '"' + ExpandConstant('{app}\setup_runtime.py') + '" --app-dir "' + ExpandConstant('{app}') + '"';
  if Exec(ExpandConstant('{app}\python\python.exe'), Params, ExpandConstant('{app}'), SW_SHOW,
          ewWaitUntilTerminated, ResultCode) and (ResultCode = 0) then
    exit;
  Log('setup_runtime.py failed with code ' + IntToStr(ResultCode));
  if not WizardSilent then
    MsgBox('RAFTcorr could not install its Python components (code ' + IntToStr(ResultCode) + ').' + #13#10 +
           'Check the internet connection, then run "Repair RAFTcorr" from the Start menu.' + #13#10 +
           'Details: ' + ExpandConstant('{app}\install.log'), mbError, MB_OK);
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
begin
  if CurUninstallStep = usPostUninstall then
    RemovePath(ExpandConstant('{app}\bin'));
end;

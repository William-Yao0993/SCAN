; Define the name of the installer
Name "ModelGUI Installer"

; Define the file that will be created
OutFile "ModelGUIInstaller.exe"

; Define the default installation directory
InstallDir $PROGRAMFILES\ModelGUI

; Request application privileges for Windows Vista and above
RequestExecutionLevel user

; Pages
Page directory   ; Allow the user to select the installation directory
Page instfiles   ; Show the installation progress

; The section is the main part of the installation
Section "Install ModelGUI"

    ; Set the output directory to the installation directory
    SetOutPath $INSTDIR

    ; Copy the main executable to the installation directory
    File "dist\ModelGUI\ModelGUI.exe"

    ; Create a shortcut to the executable in the Start Menu
    CreateDirectory "$SMPROGRAMS\ModelGUI"
    CreateShortCut "$SMPROGRAMS\ModelGUI\ModelGUI.lnk" "$INSTDIR\application.exe"

    ; Add your application to Add/Remove Programs
    WriteUninstaller "$INSTDIR\Uninstall.exe"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ModelGUI" "DisplayName" "ModelGUI"
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ModelGUI" "UninstallString" "$\"$INSTDIR\Uninstall.exe$\""
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ModelGUI" "NoModify" 1
    WriteRegDWORD HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ModelGUI" "NoRepair" 1
    WriteRegStr HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ModelGUI" "William Yao" "Danila Lab"

SectionEnd

; Uninstaller section
Section "Uninstall"
    ; Remove the installed directory and files
    Delete "$INSTDIR\application.exe"
    RMDir "$INSTDIR"

    ; Remove the Start Menu shortcut
    Delete "$SMPROGRAMS\ModelGUI\ModelGUI.lnk"
    RMDir "$SMPROGRAMS\ModelGUI"

    ; Remove the uninstaller
    Delete "$INSTDIR\Uninstall.exe"

    ; Remove the registry keys
    DeleteRegKey HKLM "Software\Microsoft\Windows\CurrentVersion\Uninstall\ModelGUI"
SectionEnd
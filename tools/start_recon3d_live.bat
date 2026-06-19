@echo off
set SCRIPT_DIR=%~dp0
powershell -ExecutionPolicy Bypass -File "%SCRIPT_DIR%start_recon3d_live.ps1" -OpenBrowser %*

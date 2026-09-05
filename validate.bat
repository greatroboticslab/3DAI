@echo off
REM Check a manifest for typos/pending rows without touching any hardware.
setlocal
cd /d "%~dp0"
set SHEET=%1
if "%SHEET%"=="" set SHEET=collection.xlsx
"%~dp0scanner_system\.venv\Scripts\python.exe" -m scanner_system.manifest validate "%SHEET%"
endlocal

@echo off
REM Run the collection manifest with the scanner's own venv, prompting per object.
REM Usage:  collect            (uses collection.xlsx)
REM         collect other.xlsx
setlocal
cd /d "%~dp0"
set SHEET=%1
if "%SHEET%"=="" set SHEET=collection.xlsx
"%~dp0scanner_system\.venv\Scripts\python.exe" -m scanner_system.manifest run "%SHEET%" --prompt
endlocal

@echo off
title 3DAI scanner
REM The one button for collecting data. Double-click it.
REM 1) checks the database, Kinect and laser board (starts the database if needed)
REM 2) scans any objects already typed into collection.xlsx
REM 3) then asks you for new objects, one at a time, and scans each one
cd /d "%~dp0"
set PY=%~dp0scanner_system\.venv\Scripts\python.exe
"%PY%" -m scanner_system.preflight
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)
echo.
"%PY%" -m scanner_system.manifest run collection.xlsx --prompt --add
echo.
echo Finished. Everything is saved in collection.xlsx and the database.
echo Double-click push_data.bat to send the data to GitHub for Dr. Zhang.
pause

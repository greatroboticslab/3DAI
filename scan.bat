@echo off
title 3DAI scanner
REM The one button for collecting data. Double-click it.
REM 1) checks the database, Kinect and laser board (starts the database if needed)
REM 2) scans any objects already typed into collection.xlsx
REM 3) then asks you for new objects, one at a time, and scans each one
cd /d "%~dp0"
set PY=%~dp0scanner_system\.venv\Scripts\python.exe
REM Optional second camera for readable laser images (see SIDE_CAMERA.md):
REM side_camera.cfg holds lines like SCANNER_LASER_CAM=1
if exist side_camera.cfg for /f "usebackq eol=# tokens=1* delims==" %%a in ("side_camera.cfg") do set %%a=%%b
"%PY%" -m scanner_system.preflight
if errorlevel 1 (
  echo.
  pause
  exit /b 1
)
echo.
REM Laser self-test: fires each laser once and confirms it lit (catches a
REM loose wire before any object is scanned). Answer 's' to skip.
"%PY%" -m scanner_system.selftest
if errorlevel 1 (
  echo.
  echo A laser did not light. Fix it, or press a key to scan anyway.
  pause
)
echo.
"%PY%" -m scanner_system.manifest run collection.xlsx --prompt --add
echo.
echo Finished. Everything is saved in collection.xlsx and the database.
echo Double-click push_data.bat to send the data to GitHub for Dr. Zhang.
pause

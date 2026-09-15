@echo off
title 3DAI push data
REM Exports the collection into the dataset\ folder and pushes it to GitHub.
cd /d "%~dp0"
set PY=%~dp0scanner_system\.venv\Scripts\python.exe
"%PY%" -m scanner_system.export_dataset dataset
if errorlevel 1 (
  echo.
  echo Export failed (is the database running? run scan.bat once to start it).
  pause
  exit /b 1
)
git add dataset collection.xlsx
git diff --cached --quiet && (
  echo Nothing new to push.
  pause
  exit /b 0
)
git commit -q -m "Update dataset export"
git push origin HEAD
if errorlevel 1 (
  echo.
  echo Push failed. Check the internet connection and try again.
  pause
  exit /b 1
)
echo.
echo Pushed. Dr. Zhang can now see the dataset folder on GitHub.
pause

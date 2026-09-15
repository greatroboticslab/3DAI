@echo off
title 3DAI online view
REM Puts the read-only scanner GUI on the internet through Tailscale so
REM Dr. Zhang (and reviewers) can browse samples, images and laser numbers
REM from anywhere. Capture and Hardware pages are hidden in this mode, so
REM nobody outside can fire the lasers.
REM
REM First time only (needs an admin account once):
REM   1. install Tailscale:  see ONLINE.md
REM   2. run this file; it prints a login link; open it and sign in
REM   3. in https://login.tailscale.com/admin/dns turn on MagicDNS + HTTPS
REM
REM Every later time: just double-click this file and leave the window open.
cd /d "%~dp0"
set PY=%~dp0scanner_system\.venv\Scripts\python.exe
set TS="C:\Program Files\Tailscale\tailscale.exe"
if not exist %TS% (
  echo Tailscale is not installed. Follow ONLINE.md first.
  pause
  exit /b 1
)

"%PY%" -m scanner_system.preflight
if errorlevel 1 ( pause & exit /b 1 )

%TS% status >nul 2>&1
if errorlevel 1 (
  echo Signing this PC into Tailscale: open the link it prints and log in.
  %TS% up
)

echo.
echo Starting the read-only GUI...
set SCANNER_GUI_READONLY=1
start "3DAI GUI (read-only)" /min "%PY%" -m streamlit run scanner_system\gui.py --server.port 8501 --server.address 127.0.0.1 --server.headless true

REM Share it. "serve" = people on our Tailscale network. To make it public
REM instead (anyone with the link), use:  tailscale funnel --bg 8501
%TS% serve --bg 8501
echo.
echo The view is online at the https address printed above.
echo Leave this window open. Close it (and the GUI window) to go offline.
pause

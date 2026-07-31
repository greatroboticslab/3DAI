@echo off
REM One-click launcher for the scanner material-recognition demo.
REM Starts MongoDB (Docker), then the Streamlit GUI, then opens the browser.

echo Starting scanner demo...

REM 1) MongoDB (mongo:7 in Docker). Ignore error if already running.
docker start scanner-mongo 2>nul || docker run -d --name scanner-mongo -p 127.0.0.1:27017:27017 mongo:7

REM 2) Environment
set SCANNER_MONGO_URL=mongodb://127.0.0.1:27017
set SCANNER_DB_NAME=scanner
set SCANNER_STORAGE_ROOT=C:\Users\Robotics_Lab\3DAI\3DAI\scanner_data

REM 3) Open the browser at the GUI
start "" http://127.0.0.1:8502

REM 4) Launch the Streamlit GUI (LAN-accessible on 0.0.0.0:8502)
cd /d C:\Users\Robotics_Lab\3DAI\3DAI
scanner_system\.venv\Scripts\python.exe -m streamlit run scanner_system\gui.py --server.address 0.0.0.0 --server.port 8502

pause

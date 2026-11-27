@echo off
chcp 65001 >nul
echo ============================================================================
echo   Moyasser AI v2.0 (Track TAX-02) - Ministry of Finance Impact Hackathon
echo   Launching on http://127.0.0.1:8080 ...
echo ============================================================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0
start "" "http://127.0.0.1:8080"
"C:\Users\zbook\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn moyasser.app:app --host 127.0.0.1 --port 8080

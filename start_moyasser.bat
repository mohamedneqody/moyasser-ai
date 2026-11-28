@echo off
chcp 65001 >nul
echo ============================================================================
echo   Moyasser AI v2.5 (Track TAX-02) - Ministry of Finance Impact Hackathon
echo   Starting server and waiting until http://127.0.0.1:8080 is ready...
echo ============================================================================
cd /d "%~dp0"
set PYTHONPATH=%~dp0
start /b "" powershell -NoProfile -Command "for($i=0;$i -lt 30;$i++){ try { $r=Invoke-WebRequest -Uri 'http://127.0.0.1:8080/api/health' -UseBasicParsing -TimeoutSec 1; if($r.StatusCode -eq 200){ Start-Process 'http://127.0.0.1:8080'; break } } catch {}; Start-Sleep -Milliseconds 400 }"
"C:\Users\zbook\AppData\Local\Programs\Python\Python312\python.exe" -m uvicorn moyasser.app:app --host 127.0.0.1 --port 8080

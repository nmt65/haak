@echo off
chcp 65001 >nul
cd /d "%~dp0neurovision_benchmark"
if not exist ".venv\Scripts\python.exe" (
  echo Mai intai ruleaza 01_INSTALARE_BACKEND_WINDOWS.bat
  pause
  exit /b 1
)
echo Pornesc backend-ul pe http://127.0.0.1:8000
".venv\Scripts\python.exe" -m uvicorn server.app:app --host 127.0.0.1 --port 8000
if errorlevel 1 (
  echo Backend-ul s-a oprit cu eroare. Verifica mesajul de mai sus.
  pause
)

@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0neurovision_benchmark"
if not exist ".venv\Scripts\python.exe" (
  echo [1/2] Creez mediul Python pentru backend...
  py -3.11 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
  if not exist ".venv\Scripts\python.exe" python -m venv .venv
)
if not exist ".venv\Scripts\python.exe" (
  echo EROARE: Instaleaza Python 3.11 si bifeaza Add Python to PATH.
  pause
  exit /b 1
)
echo [2/2] Instalez dependentele Python. Prima instalare poate descarca PyTorch.
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt -r requirements-server.txt
if errorlevel 1 (
  echo Instalarea a esuat. Verifica erorile de mai sus.
  pause
  exit /b 1
)
echo INSTALARE COMPLETA. Deschide 02_PORNESTE_BACKEND_WINDOWS.bat
pause

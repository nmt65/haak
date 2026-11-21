@echo off
chcp 65001 >nul
cd /d "%~dp0"
where npm >nul 2>nul
if errorlevel 1 (
  echo EROARE: Instaleaza Node.js 20 sau mai nou de pe nodejs.org
  pause
  exit /b 1
)
if not exist "node_modules" (
  echo Instalez dependentele frontend...
  call npm ci
  if errorlevel 1 (
    echo Instalarea a esuat. Verifica erorile de mai sus.
    pause
    exit /b 1
  )
)
echo Deschide http://127.0.0.1:5173 in browser.
call npm run dev -- --host 127.0.0.1
pause

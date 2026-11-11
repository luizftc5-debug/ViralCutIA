@echo off
rem ViralCut: starts the API and the site, then opens the browser. Double-click to run (Windows, no Docker).
rem First time only: install the dependencies, see the Windows section of README.md.
setlocal
set "ROOT=%~dp0.."

if not exist "%ROOT%\apps\api\.venv\Scripts\uvicorn.exe" (
  echo The API is not installed yet. Follow the Windows section of README.md first.
  pause
  exit /b 1
)
if not exist "%ROOT%\apps\web\node_modules" (
  echo The site is not installed yet. Run "npm install" inside apps\web first.
  pause
  exit /b 1
)

set "VIRALCUT_EMBEDDED_WORKER=true"
start "ViralCut API" /D "%ROOT%\apps\api" cmd /k .venv\Scripts\uvicorn viralcut.api.app:app --host 127.0.0.1 --port 8000
start "ViralCut Site" /D "%ROOT%\apps\web" cmd /k npm run dev
timeout /t 10 /nobreak >nul
start http://localhost:3000
endlocal

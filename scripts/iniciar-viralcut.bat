@echo off
rem ViralCut: starts the API and the site, then opens the browser. Double-click to run (Windows, no Docker).
rem First time only: install the dependencies, see docs\COMECAR.md (or the Windows section of README.md).
rem After pulling a new version, run atualizar-viralcut.bat once.
setlocal
set "ROOT=%~dp0.."

if not exist "%ROOT%\apps\api\.venv\Scripts\uvicorn.exe" (
  echo The API is not installed yet. Follow docs\COMECAR.md first.
  pause
  exit /b 1
)
if not exist "%ROOT%\apps\web\node_modules" (
  echo The site is not installed yet. Run atualizar-viralcut.bat first.
  pause
  exit /b 1
)
rem without ffmpeg the worker cannot run and every video waits forever in the queue
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found. Install it with:  winget install Gyan.FFmpeg
  echo Then close this window and open iniciar-viralcut.bat again.
  pause
  exit /b 1
)
if not exist "%ROOT%\apps\web\.next\BUILD_ID" (
  echo Building the site once ^(about a minute^)...
  pushd "%ROOT%\apps\web"
  call npm run build
  popd
)

rem everything ViralCut writes stays inside this folder: data and temp files in apps\api\data,
rem AI models in models\ (set by the API itself, see apps\api\viralcut\config.py)
rem the worker runs inside the API; ECO / BALANCED / TURBO is chosen per video in the site
set "VIRALCUT_EMBEDDED_WORKER=true"
start "ViralCut API" /D "%ROOT%\apps\api" cmd /k .venv\Scripts\uvicorn viralcut.api.app:app --host 127.0.0.1 --port 8000
rem production build: lighter on CPU and RAM than the development server
start "ViralCut Site" /D "%ROOT%\apps\web" cmd /k npm run start -- -p 3000
timeout /t 8 /nobreak >nul
start http://localhost:3000
endlocal

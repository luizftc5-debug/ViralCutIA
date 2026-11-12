@echo off
rem ViralCut: starts the API and the site, then opens the browser. Double-click to run (Windows, no Docker).
rem First time only: install the dependencies, see docs\COMECAR.md (or the Windows section of README.md).
rem After pulling a new version, run atualizar-viralcut.bat once.
rem The project folder can be moved or renamed: this script detects it and repairs the install.
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "PY=%ROOT%\apps\api\.venv\Scripts\python.exe"
set "MARKER=%ROOT%\apps\api\.venv\viralcut-path.txt"

if not exist "%PY%" goto :no_api
if not exist "%ROOT%\apps\web\node_modules" goto :no_web

rem the Python environment must still work (it stops working if the base Python was uninstalled)
"%PY%" -c "import uvicorn" >nul 2>&1 || goto :broken_api

rem folder moved or renamed since the last start? re-point the install to the new place
set "OLDROOT="
if exist "%MARKER%" set /p OLDROOT=<"%MARKER%"
if /i not "%OLDROOT%"=="%ROOT%" call :relocate || goto :broken_api

rem without ffmpeg the worker cannot run and every video waits forever in the queue
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo ffmpeg was not found. Install it with:  winget install Gyan.FFmpeg
  echo Then close this window and open iniciar-viralcut.bat again.
  pause
  exit /b 1
)
if not exist "%ROOT%\apps\web\.next\BUILD_ID" (
  echo Building the site once, about a minute...
  pushd "%ROOT%\apps\web"
  call npm run build
  popd
)
if not exist "%ROOT%\apps\web\.next\BUILD_ID" goto :no_build

rem something left over on port 8000 (an API from the old folder, for example) would block the new one
netstat -ano | findstr /r /c:":8000 .*LISTENING" /c:":8000 .*ESCUTANDO" >nul && (
  echo Port 8000 is already in use. Close the old "ViralCut API" window, or restart the PC, and try again.
  pause
  exit /b 1
)

rem everything ViralCut writes stays inside this folder: data and temp files in apps\api\data,
rem AI models in models\ (set by the API itself, see apps\api\viralcut\config.py)
rem the worker runs inside the API; ECO / BALANCED / TURBO is chosen per video in the site
set "VIRALCUT_EMBEDDED_WORKER=true"
rem "python -m uvicorn" instead of uvicorn.exe: the .exe stores the old folder's path and breaks after a move
start "ViralCut API" /D "%ROOT%\apps\api" cmd /k ""%PY%" -m uvicorn viralcut.api.app:app --host 127.0.0.1 --port 8000"
rem production build: lighter on CPU and RAM than the development server
start "ViralCut Site" /D "%ROOT%\apps\web" cmd /k npm run start -- -p 3000
timeout /t 8 /nobreak >nul
start http://localhost:3000
endlocal
exit /b 0

:relocate
if not defined OLDROOT goto :relocate_go
echo The project folder changed:
echo   from %OLDROOT%
echo   to   %ROOT%
:relocate_go
echo Pointing the API and the site to this folder, takes a minute...
"%PY%" -m pip install --no-deps -e "%ROOT%\apps\api" >nul || exit /b 1
rem the site build can keep paths of the old folder: rebuild it below
if exist "%ROOT%\apps\web\.next" rmdir /s /q "%ROOT%\apps\web\.next"
> "%MARKER%" echo %ROOT%
exit /b 0

:no_api
echo The API is not installed yet. Run atualizar-viralcut.bat first (see docs\COMECAR.md).
pause
exit /b 1

:no_web
echo The site is not installed yet. Run atualizar-viralcut.bat first.
pause
exit /b 1

:broken_api
echo The Python environment in apps\api\.venv is broken.
echo Delete the folder apps\api\.venv and run atualizar-viralcut.bat.
pause
exit /b 1

:no_build
echo The site build failed, see the messages above. Run atualizar-viralcut.bat.
pause
exit /b 1

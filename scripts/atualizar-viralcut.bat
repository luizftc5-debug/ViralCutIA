@echo off
rem ViralCut: install or update the dependencies and rebuild the site. Run once after each update.
setlocal
set "ROOT=%~dp0.."
rem download caches of pip and npm stay inside the project folder, not the user profile on C:
set "PIP_CACHE_DIR=%ROOT%\.cache\pip"
set "npm_config_cache=%ROOT%\.cache\npm"

if not exist "%ROOT%\apps\api\.venv\Scripts\python.exe" (
  echo Creating the Python environment...
  py -3.13 -m venv "%ROOT%\apps\api\.venv" || python -m venv "%ROOT%\apps\api\.venv" || goto :fail
)
echo Installing the API...
"%ROOT%\apps\api\.venv\Scripts\python.exe" -m pip install --upgrade pip >nul
"%ROOT%\apps\api\.venv\Scripts\python.exe" -m pip install -e "%ROOT%\apps\api[llm]" || goto :fail

echo Installing and building the site...
pushd "%ROOT%\apps\web"
call npm install --no-audit --no-fund || (popd & goto :fail)
call npm run build || (popd & goto :fail)
popd

echo.
echo Done. Start ViralCut with iniciar-viralcut.bat
pause
exit /b 0

:fail
echo.
echo Something failed above. Check that Python 3.11-3.13, Node 20+ and ffmpeg are installed.
pause
exit /b 1

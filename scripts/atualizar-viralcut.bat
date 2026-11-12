@echo off
rem ViralCut: install or update the dependencies and rebuild the site. Run once after each update.
rem Also repairs the install after the project folder is moved or renamed.
setlocal
for %%I in ("%~dp0..") do set "ROOT=%%~fI"
set "VENV=%ROOT%\apps\api\.venv"
set "PY=%VENV%\Scripts\python.exe"
rem download caches of pip and npm stay inside the project folder, not the user profile on C:
set "PIP_CACHE_DIR=%ROOT%\.cache\pip"
set "npm_config_cache=%ROOT%\.cache\npm"

rem a Python environment whose base Python is gone cannot be repaired: start it again
if exist "%PY%" (
  "%PY%" -c "import sys" >nul 2>&1 || (
    echo The Python environment is broken, recreating it...
    rmdir /s /q "%VENV%"
  )
)
if not exist "%PY%" (
  echo Creating the Python environment...
  py -3.13 -m venv "%VENV%" || python -m venv "%VENV%" || goto :fail
)
echo Installing the API...
"%PY%" -m pip install --upgrade pip >nul
rem the editable install records the folder path: reinstalling it also fixes a moved folder
"%PY%" -m pip install -e "%ROOT%\apps\api[llm]" || goto :fail
> "%VENV%\viralcut-path.txt" echo %ROOT%

echo Installing and building the site...
pushd "%ROOT%\apps\web"
if exist .next rmdir /s /q .next
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

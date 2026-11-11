# Start ViralCut AI on Windows without Docker.
#   API with an embedded worker on http://localhost:8000 (own window)
#   Web app on http://localhost:3000 (this window; Ctrl+C stops both)
#
# Run from the repository root:
#   powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
#
# Needs: Python 3.11+ (py launcher or python on PATH), Node.js 20+, and a full ffmpeg build
# on PATH (libx264 + libass). scripts\doctor.py checks all of this first.

$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$Api = Join-Path $Root 'apps\api'
$Web = Join-Path $Root 'apps\web'
$VenvPy = Join-Path $Api '.venv\Scripts\python.exe'

function Find-Python {
    # Returns the command (as an array) of the first Python 3.11+ interpreter found.
    # Only 3.11-3.13: onnxruntime (a faster-whisper dependency) has no wheels for newer Pythons yet.
    $candidates = @(@('py', '-3.12'), @('py', '-3.13'), @('py', '-3.11'), @('python'))
    foreach ($c in $candidates) {
        $exe = $c[0]
        $rest = @($c | Select-Object -Skip 1)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $v = & $exe @rest -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0 -and $v -and ([version]$v -ge [version]'3.11') -and ([version]$v -lt [version]'3.14')) {
            return , $c
        }
    }
    return $null
}

$PyCmd = Find-Python
if (-not $PyCmd) {
    Write-Host 'Python 3.11, 3.12 or 3.13 was not found (3.14+ is not supported yet).' -ForegroundColor Red
    Write-Host 'Install Python 3.12 with: winget install --id Python.Python.3.12 -e' -ForegroundColor Yellow
    Write-Host 'Then open a new PowerShell window and run this script again.'
    exit 1
}
$PyExe = $PyCmd[0]
$PyArgs = @($PyCmd | Select-Object -Skip 1)

# 1. Check prerequisites (ffmpeg with libx264 and libass, Node.js, ...).
& $PyExe @PyArgs (Join-Path $Root 'scripts\doctor.py')
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Fix the items marked [FAIL] above, then run this script again.' -ForegroundColor Red
    exit 1
}

# 2. Python environment for the API (first run only).
if (-not (Test-Path $VenvPy)) {
    Write-Host 'Creating the Python environment (first run, takes a few minutes)...' -ForegroundColor Cyan
    & $PyExe @PyArgs -m venv (Join-Path $Api '.venv')
    & $VenvPy -m pip install --upgrade pip
    & $VenvPy -m pip install -e "${Api}[dev,llm]"
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'pip install failed. Delete apps\api\.venv and run this script again.' -ForegroundColor Red
        exit 1
    }
}

# 3. Web dependencies (first run only). npm.cmd avoids PowerShell's execution-policy block on npm.ps1.
if (-not (Test-Path (Join-Path $Web 'node_modules'))) {
    Write-Host 'Installing web dependencies (first run)...' -ForegroundColor Cyan
    Push-Location $Web
    npm.cmd install --no-audit --no-fund
    $npmExit = $LASTEXITCODE
    Pop-Location
    if ($npmExit -ne 0) {
        Write-Host 'npm install failed.' -ForegroundColor Red
        exit 1
    }
}

# 4. Runtime settings. The root .env (copied from .env.example) is read by the API automatically.
$env:VIRALCUT_DATA_DIR = Join-Path $Root 'data'
$env:VIRALCUT_EMBEDDED_WORKER = 'true'
$env:PYTHONIOENCODING = 'utf-8'

# 5. API. If port 8000 is taken, an old API (often from a previous run) is still alive.
if (Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue) {
    Write-Host 'Port 8000 is already in use (an old API is probably still running).' -ForegroundColor Red
    Write-Host 'Stop it with this command, then run this script again:' -ForegroundColor Yellow
    Write-Host '  Get-NetTCPConnection -LocalPort 8000 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }'
    exit 1
}

# The API log goes to data\api.err.log (the window is hidden); it is shown here if the API fails.
New-Item -ItemType Directory -Force -Path $env:VIRALCUT_DATA_DIR | Out-Null
$ApiOut = Join-Path $env:VIRALCUT_DATA_DIR 'api.out.log'
$ApiErr = Join-Path $env:VIRALCUT_DATA_DIR 'api.err.log'
$ApiProc = Start-Process -FilePath $VenvPy `
    -ArgumentList '-m', 'uvicorn', 'viralcut.api.app:app', '--host', '127.0.0.1', '--port', '8000' `
    -WorkingDirectory $Api -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $ApiOut -RedirectStandardError $ApiErr

function Stop-Api {
    # taskkill /T also stops the child python process that the venv launcher starts.
    if ($ApiProc -and -not $ApiProc.HasExited) {
        taskkill.exe /T /F /PID $ApiProc.Id | Out-Null
    }
}

# Wait until the API answers (the first start can take 20-60 s: imports, antivirus, OneDrive).
Write-Host 'Starting the API...' -ForegroundColor Cyan
$ready = $false
for ($i = 0; $i -lt 120; $i++) {
    if ($ApiProc.HasExited) { break }
    try {
        $r = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:8000/docs' -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { }
    Start-Sleep -Seconds 1
}
if (-not $ready) {
    Write-Host 'The API did not start. Last lines of its log:' -ForegroundColor Red
    foreach ($f in @($ApiErr, $ApiOut)) {
        if (Test-Path $f) { Get-Content $f -Tail 40 }
    }
    Write-Host "Full logs: $ApiErr" -ForegroundColor Yellow
    Stop-Api
    exit 1
}

Write-Host ''
Write-Host 'API is up. Open http://localhost:3000 in your browser (first job downloads the speech model, ~480 MB).' -ForegroundColor Green
Write-Host 'API log: data\api.err.log. Press Ctrl+C here to stop everything.'
Write-Host ''

try {
    Push-Location $Web
    npm.cmd run dev
} finally {
    Pop-Location
    Stop-Api
}

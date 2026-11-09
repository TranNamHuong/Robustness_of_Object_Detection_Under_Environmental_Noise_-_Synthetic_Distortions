param(
  [switch]$NoBrowser
)

$ErrorActionPreference = 'Stop'
$projectRoot = $PSScriptRoot
$runtimeDir = Join-Path $projectRoot '.website_runtime'
$logDir = Join-Path $runtimeDir 'logs'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$pythonExe = Join-Path $projectRoot '.venv\Scripts\python.exe'
$nodeExe = (Get-Command node -ErrorAction Stop).Source
$viteEntry = Join-Path $projectRoot 'web\node_modules\vite\bin\vite.js'

if (-not (Test-Path -LiteralPath $pythonExe)) { throw "Project Python environment missing: $pythonExe" }
if (-not (Test-Path -LiteralPath $viteEntry)) { throw "Frontend dependencies missing. Install web dependencies first; expected: $viteEntry" }

function Test-LocalEndpoint([string]$url) {
  try { Invoke-RestMethod -Uri $url -TimeoutSec 2 | Out-Null; return $true } catch { return $false }
}

if (-not (Test-LocalEndpoint 'http://127.0.0.1:8000/api/health')) {
  $api = Start-Process -FilePath $pythonExe -ArgumentList @('-m','uvicorn','server.robustness_api.main:app','--host','127.0.0.1','--port','8000') -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'api.out.log') -RedirectStandardError (Join-Path $logDir 'api.err.log')
  Set-Content -LiteralPath (Join-Path $runtimeDir 'api.pid') -Value $api.Id -NoNewline
  $apiReady = $false
  for ($i = 0; $i -lt 60; $i++) {
    Start-Sleep -Milliseconds 1000
    if (Test-LocalEndpoint 'http://127.0.0.1:8000/api/health') { $apiReady = $true; break }
    if ($api.HasExited) { break }
  }
  if (-not $apiReady) {
    $detail = Get-Content -LiteralPath (Join-Path $logDir 'api.err.log') -Raw -ErrorAction SilentlyContinue
    throw "Backend did not become ready. Log: $detail"
  }
}

if (-not (Test-LocalEndpoint 'http://127.0.0.1:5173')) {
  $web = Start-Process -FilePath $nodeExe -ArgumentList @($viteEntry,'--host','127.0.0.1','--port','5173') -WorkingDirectory (Join-Path $projectRoot 'web') -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $logDir 'web.out.log') -RedirectStandardError (Join-Path $logDir 'web.err.log')
  Set-Content -LiteralPath (Join-Path $runtimeDir 'web.pid') -Value $web.Id -NoNewline
  $webReady = $false
  for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 500
    if (Test-LocalEndpoint 'http://127.0.0.1:5173') { $webReady = $true; break }
  }
  if (-not $webReady) { throw "Frontend did not become ready. See $logDir\web.err.log" }
}

Write-Host 'Object Detection Robustness Lab is ready:' -ForegroundColor Green
Write-Host '  Website: http://127.0.0.1:5173'
Write-Host '  API docs: http://127.0.0.1:8000/docs'
Write-Host "  Logs: $logDir"
if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:5173' }

$ErrorActionPreference = 'SilentlyContinue'
$runtimeDir = Join-Path $PSScriptRoot '.website_runtime'
foreach ($name in @('web.pid','api.pid')) {
  $pidFile = Join-Path $runtimeDir $name
  if (Test-Path -LiteralPath $pidFile) {
    $processId = [int](Get-Content -LiteralPath $pidFile -Raw)
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if ($process) { Stop-Process -Id $processId -Force }
    Remove-Item -LiteralPath $pidFile -Force
  }
}
Write-Host 'Stopped local website processes recorded by this project.'

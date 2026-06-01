$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonPath = Join-Path $projectRoot ".venv311\Scripts\python.exe"
$appPath = Join-Path $projectRoot "app.py"

if (-not (Test-Path $pythonPath)) {
    throw "Missing Python interpreter: $pythonPath"
}

Set-Location $projectRoot
& $pythonPath $appPath

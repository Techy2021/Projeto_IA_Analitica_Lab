$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Ambiente virtual nao encontrado em $Python"
}

Set-Location $ProjectRoot
& $Python -m uvicorn api.model_api:app --host 127.0.0.1 --port 8000

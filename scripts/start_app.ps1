$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Ambiente virtual nao encontrado em $Python"
}

Set-Location $ProjectRoot
& $Python -m streamlit run app/main.py --server.address localhost --server.port 8501

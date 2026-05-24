param(
    [int]$Port = 3000
)

$ErrorActionPreference = "Stop"

$Node20Path = "D:\tools\node-v20.20.2-win-x64"
$NpmGlobalPath = "D:\npm-global"
$Python311 = "C:\Users\Techy\AppData\Local\Programs\Python\Python311\python.exe"

Write-Host "Preparando ambiente do Flowise..." -ForegroundColor Cyan

if (-not (Test-Path $Node20Path)) {
    Write-Host "Node 20 portatil nao encontrado em $Node20Path" -ForegroundColor Red
    Write-Host "Instale o Node 20 ou veja docs\flowise_setup_windows.md." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path $Python311)) {
    Write-Host "Python 3.11 nao encontrado em $Python311" -ForegroundColor Red
    exit 1
}

New-Item -ItemType Directory -Path "D:\npm-cache" -Force | Out-Null
New-Item -ItemType Directory -Path $NpmGlobalPath -Force | Out-Null

$env:Path = "$NpmGlobalPath;$Node20Path;$env:Path"
$env:npm_config_python = $Python311
$env:PYTHON = $Python311
$env:PORT = "$Port"

Write-Host "Node: $(node -v)" -ForegroundColor Green
Write-Host "npm:  $(npm -v)" -ForegroundColor Green
Write-Host "Porta: $Port" -ForegroundColor Green

$FlowiseCmd = Join-Path $NpmGlobalPath "flowise.cmd"

if (-not (Test-Path $FlowiseCmd)) {
    Write-Host "Flowise ainda nao esta instalado em $NpmGlobalPath." -ForegroundColor Red
    Write-Host "Instale com: npm install -g flowise" -ForegroundColor Yellow
    Write-Host "Se falhar em better-sqlite3/node-gyp, instale Visual Studio Build Tools com Desktop development with C++." -ForegroundColor Yellow
    exit 1
}

Write-Host "Iniciando Flowise em http://localhost:$Port ..." -ForegroundColor Cyan
flowise start --PORT=$Port

# Flowise no Windows

## Diagnostico

Projeto: `D:\Projeto_IA_Analitica_Lab`

Ambiente validado em 2026-05-23:

- Windows 10
- Python 3.11 em `C:\Users\Techy\AppData\Local\Programs\Python\Python311\python.exe`
- API FastAPI funcionando em `http://localhost:8000`
- npm configurado para:
  - cache: `D:\npm-cache`
  - prefix: `D:\npm-global`
- Node instalado no sistema:
  - `node v24.15.0`
  - `npm 11.12.1`
  - caminho: `C:\Program Files\nodejs\node.exe`
- NVM nao esta instalado: `nvm` nao foi reconhecido no PowerShell.
- Espaco em disco:
  - `C:` com pouco espaco livre, cerca de 6 GB no momento do diagnostico.
  - `D:` com espaco suficiente, cerca de 88 GB livres.

## Erro encontrado

A instalacao global com `npm install -g flowise` falhou em dependencias nativas dentro de:

```powershell
D:\npm-global\node_modules\flowise\node_modules
```

O erro original ocorreu em `better-sqlite3` com Node 24:

```text
No prebuilt binaries found (target=24.15.0 runtime=node arch=x64 platform=win32)
could not find a version of Visual Studio 2017 or newer to use
```

Depois da falha, `flowise start` passou a retornar:

```text
Cannot find module 'D:\npm-global\node_modules\flowise\bin\run'
```

Isso indica uma instalacao parcial/incompleta.

## Causa provavel

O Flowise depende de pacotes Node com codigo nativo, especialmente `better-sqlite3`. Quando nao existe binario pre-compilado compativel com a versao do Node em uso, o npm tenta compilar localmente via `node-gyp`.

O `node-gyp` encontrou corretamente o Python 3.11, mas nao encontrou um toolset C++ completo do Visual Studio. Na maquina existe uma instalacao parcial de Build Tools em:

```text
C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools
```

Porem o log informa que falta o VC++ toolset:

```text
found "Visual Studio C++ core features"
missing any VC++ toolset
could not find a version of Visual Studio 2017 or newer to use
```

Tambem foi testado Node 20 portatil em:

```text
D:\tools\node-v20.20.2-win-x64
```

Com:

```text
node v20.20.2
npm 10.8.2
```

Mesmo com Node 20, a instalacao falhou porque `better-sqlite3` nao encontrou binario pre-compilado para `v20.20.2` e tentou compilar localmente. Portanto, neste ambiente a dependencia obrigatoria faltante e o toolset C++ do Visual Studio Build Tools.

## Comandos executados

Diagnostico:

```powershell
node -v
npm -v
where.exe node
where.exe npm
npm config get cache
npm config get prefix
npm config list
py -3.11 -c "import sys; print(sys.executable)"
Get-PSDrive -PSProvider FileSystem
nvm version
```

Preparacao do npm em `D:`:

```powershell
New-Item -ItemType Directory -Path D:\npm-cache -Force
New-Item -ItemType Directory -Path D:\npm-global -Force
npm config set cache D:\npm-cache --global
npm config set prefix D:\npm-global --global
```

Node 20 portatil:

```powershell
New-Item -ItemType Directory -Path D:\tools -Force
# Baixado de https://nodejs.org/dist/v20.20.2/node-v20.20.2-win-x64.zip
$env:Path = "D:\npm-global;D:\tools\node-v20.20.2-win-x64;$env:Path"
node -v
npm -v
```

Limpeza da instalacao incompleta:

```powershell
taskkill /F /IM node.exe
Remove-Item D:\npm-global\node_modules\flowise -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item D:\npm-global\flowise -Force -ErrorAction SilentlyContinue
Remove-Item D:\npm-global\flowise.cmd -Force -ErrorAction SilentlyContinue
Remove-Item D:\npm-global\flowise.ps1 -Force -ErrorAction SilentlyContinue
Remove-Item D:\npm-cache\_npx -Recurse -Force -ErrorAction SilentlyContinue
npm cache clean --force
```

Configuracao do Python para `node-gyp`:

```powershell
$env:npm_config_python = "C:\Users\Techy\AppData\Local\Programs\Python\Python311\python.exe"
$env:PYTHON = "C:\Users\Techy\AppData\Local\Programs\Python\Python311\python.exe"
```

Tentativas de instalacao:

```powershell
npm install -g flowise
npm install -g flowise --omit=optional
```

Resultado: ambas falharam por falta de Visual Studio Build Tools com toolset C++.

## Solucao adotada

Foi mantida uma instalacao portatil do Node 20 em `D:\tools\node-v20.20.2-win-x64`, para evitar alterar o ambiente Python e reduzir impacto no Windows.

O Flowise ainda nao foi instalado porque falta instalar/corrigir o Visual Studio Build Tools com suporte C++.

Instale ou modifique o Visual Studio Build Tools incluindo o workload:

```text
Desktop development with C++
```

Componentes recomendados:

- MSVC v143 ou toolset C++ mais recente disponivel no instalador
- Windows 10 SDK ou Windows 11 SDK
- C++ CMake tools for Windows

Depois de instalar esses componentes, reinicie o computador e rode:

```powershell
cd D:\Projeto_IA_Analitica_Lab
$env:Path = "D:\npm-global;D:\tools\node-v20.20.2-win-x64;$env:Path"
$env:npm_config_python = "C:\Users\Techy\AppData\Local\Programs\Python\Python311\python.exe"
$env:PYTHON = "C:\Users\Techy\AppData\Local\Programs\Python\Python311\python.exe"
npm install -g flowise
```

## Como iniciar o Flowise

Depois que `npm install -g flowise` concluir com sucesso:

```powershell
cd D:\Projeto_IA_Analitica_Lab
.\scripts\start_flowise.ps1
```

O script inicia na porta 3000. Acesse:

```text
http://localhost:3000
```

Se a porta 3000 estiver ocupada:

```powershell
.\scripts\start_flowise.ps1 -Port 3001
```

E acesse:

```text
http://localhost:3001
```

## Como conectar o Flowise a API FastAPI

Com a API em execucao em `http://localhost:8000`, crie ferramentas HTTP no Flowise para chamar os endpoints do projeto.

Ferramenta `prever_farelo_soja`:

- Metodo: `POST`
- URL: `http://localhost:8000/prever`
- Headers:
  - `Content-Type: application/json`
- Body: JSON no formato esperado pelo endpoint `/prever`.
- Uso: previsao do valor/resultado de farelo de soja com base nas variaveis do modelo.

Ferramenta `consultar_dados_laboratorio`:

- Metodo: `GET` ou `POST`, conforme endpoint criado/existente na FastAPI.
- URL base: `http://localhost:8000`
- Uso: consultar dados de laboratorio ou dados historicos carregados no DuckDB.

No Flowise, use um Agent/Chatflow com ferramentas HTTP ou Custom Tool para chamar esses endpoints. Primeiro valide cada chamada isoladamente, depois conecte ao agente.

## Plano alternativo se Flowise nao rodar

Se o Flowise continuar bloqueado por dependencias nativas no Windows:

1. Rodar Flowise via Docker, mantendo a API FastAPI no Windows.
2. Rodar Flowise em WSL2 com Node LTS e toolchain Linux.
3. Criar uma camada de chat simples em Streamlit usando a API FastAPI diretamente.
4. Criar um microservico de agente em Python, evitando dependencias Node nativas.

Para este projeto, a alternativa mais rapida sem Node nativo e um chat Streamlit chamando `http://localhost:8000/prever`.

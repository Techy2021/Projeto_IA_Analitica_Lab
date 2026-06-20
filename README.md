# Projeto IA Analitica Lab

Plataforma local para analise de dados laboratoriais, treinamento de modelos
preditivos, consultas numericas com DuckDB, agentes de IA com LLM local ou online e recuperacao
de conhecimento por RAG.

## Objetivo

O projeto centraliza dados laboratoriais de soja, farelo e casca para:

- importar e explorar datasets;
- consultar dados com SQL;
- treinar modelos com FLAML AutoML;
- executar previsoes com o modelo salvo;
- responder perguntas numericas diretamente pelo DuckDB;
- responder perguntas conceituais com agentes CrewAI e um provedor de LLM configuravel;
- indexar documentos e recuperar trechos com ChromaDB;
- registrar traces, metricas e resultados de testes.

O sistema e uma ferramenta de apoio. Ele nao substitui metodos oficiais, validacao
laboratorial, revisao humana ou criterios formais de liberacao de lotes.

## Tecnologias

| Camada | Tecnologia |
|---|---|
| Interface | Streamlit |
| API | FastAPI e Uvicorn |
| Banco analitico | DuckDB |
| Manipulacao de dados | pandas e NumPy |
| AutoML | FLAML e scikit-learn |
| Persistencia do modelo | joblib |
| Agentes | CrewAI |
| LLM | Ollama local, OpenAI API, Google Gemini ou endpoint compativel via LiteLLM |
| RAG e vetores | ChromaDB e embeddings Ollama |
| Graficos | Plotly e Matplotlib |
| Testes de IA | unittest, scripts golden dataset e DeepEval opcional |

## Arquitetura

```text
app/                    Interface Streamlit, configuracao e componentes
ai/
  intent_router.py      Classificacao de intencao e extracao de entidades
  llm_provider.py       Selecao central, teste e chamada segura dos provedores LLM
  numeric_query_engine.py Consultas estatisticas seguras no DuckDB
  agentes/              Orquestracao CrewAI, ferramentas e respostas
  prompts/              Prompts compartilhados
  rag/                  Loader, indexacao, busca e roteamento numerico
  modeling/             Treinamento FLAML e inferencia
api/                    Endpoints FastAPI
data/
  raw/                  Dados brutos
  processed/            Dados tratados
  examples/             Datasets de exemplo
  knowledge_base/       Documentos usados pelo RAG
  vectorstore/          Persistencia do ChromaDB
database/               Conexao e consultas DuckDB
models/                 Modelo, metadados e copia das metricas
tests/
  unit/                 Testes minimos, rapidos e deterministas
  ai/                   Avaliacao de qualidade das respostas
  integration/          Testes com API, agentes e Ollama
reports/                Metricas, traces, logs e resultados
docs/                   Documentacao tecnica e didatica
scripts/                Scripts auxiliares para Windows
```

Detalhes adicionais:

- [Visao geral](docs/01_visao_geral.md)
- [Fluxo do sistema](docs/02_fluxo_do_sistema.md)
- [Modelo preditivo](docs/03_modelo_preditivo.md)
- [Agentes de IA](docs/04_agentes_ia.md)
- [Testes e validacao](docs/05_testes_e_validacao.md)
- [Limitacoes](docs/06_limitacoes.md)
- [Guia para desenvolvedores](docs/07_guia_para_novos_desenvolvedores.md)

## Requisitos

- Windows com PowerShell;
- Python 3.11 recomendado;
- Ollama para uso local do chat e para embeddings, quando aplicavel;
- ou uma chave de API de provedor online para o chat em deploy;
- ambiente virtual Python.

## Instalacao

O projeto possui dois arquivos de dependencias:

- `requirements.txt`: ambiente completo para desenvolvimento, testes, RAG,
  avaliacao de IA e execucao local;
- `requirements-deploy.txt`: ambiente reduzido para a aplicacao Streamlit em
  Docker e no Render.

Para desenvolvimento local:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Para reproduzir apenas o ambiente de producao:

```powershell
python -m pip install -r requirements-deploy.txt
```

O `.env.example` nao possui chaves reais. Ajuste o provedor, o modelo e os caminhos
somente quando necessario. Nunca versione o arquivo `.env`.

## Configuracao do provedor de IA

O sistema oferece tres provedores principais para o chat:

- Ollama local;
- OpenAI API;
- Google Gemini via Google AI Studio.

Tambem existe suporte a endpoints customizados compativeis com a API da OpenAI.
O chat usa `ai/llm_provider.py` como ponto central. A configuracao aplicada na
tela **Configuracoes** vale apenas para a sessao atual do Streamlit. Ela nao
altera o `.env` e a chave digitada permanece somente em memoria.

A precedencia e:

1. configuracao aplicada na sessao do Streamlit;
2. variaveis de ambiente;
3. valores padrao seguros.

### Ollama local

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3-vl:4b
OLLAMA_BASE_URL=http://localhost:11434
```

Inicie o Ollama, confirme que o modelo foi baixado e execute normalmente a API
e o Streamlit. Para o alias `qwen3-vl:4b`, o chat usa automaticamente a variante
`qwen3-vl:4b-instruct`. Outros exemplos sao `gemma3:4b` e `llama3.2:3b`.

### OpenAI API

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
```

Defina `OPENAI_API_KEY` no gerenciador de segredos do ambiente de deploy.
`OPENAI_BASE_URL` deve permanecer vazio para o endpoint oficial. A chave nao
deve ser colocada no Dockerfile, na imagem, no repositorio ou em logs.

### Google Gemini via Google AI Studio

Crie uma chave no Google AI Studio e configure:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
```

O projeto usa o LiteLLM com o formato `gemini/gemini-2.5-flash`, necessário
para acessar a Gemini API com uma chave do Google AI Studio. Mantenha
`GEMINI_API_KEY` somente no ambiente local ou no gerenciador de segredos do
Docker/Render. Se a chave estiver vazia, a interface informa
`Chave GEMINI_API_KEY não configurada.` sem interromper a aplicação.

### Outro endpoint via API

Endpoints compativeis com a API da OpenAI podem ser usados por meio do LiteLLM:

```env
LLM_PROVIDER=custom
OTHER_LLM_API_KEY=
OTHER_LLM_MODEL=nome-do-modelo
OTHER_LLM_BASE_URL=https://provedor.exemplo/v1
```

Na interface, abra **Configuracoes > Configuracao do provedor de IA**, selecione
`ollama`, `openai` ou `gemini`, informe os campos desejados e use
**Testar conexao com o provedor de IA**. Os valores da interface têm prioridade
durante a sessão; campos não preenchidos usam as variáveis de ambiente como
fallback. As chaves são mantidas somente em `st.session_state`, não são salvas
em arquivo, banco ou logs. O teste não exibe nem registra a chave.

## Execucao

Inicie a API:

```powershell
.\scripts\start_api.ps1
```

Em outro terminal, inicie a interface:

```powershell
.\scripts\start_app.ps1
```

- Interface: `http://localhost:8501`
- API: `http://localhost:8000`
- Documentacao da API: `http://localhost:8000/docs`
- Health check: `http://localhost:8000/health`

Comandos diretos:

```powershell
python -m uvicorn api.model_api:app --host 127.0.0.1 --port 8000
python -m streamlit run app/main.py --server.address localhost --server.port 8501
```

## Dados e DuckDB

No ambiente atual, o banco principal e `data/lab_ia.duckdb`. O `.env.example`
aponta novos ambientes para `data/processed/laboratorio.duckdb`; ajuste
`DATABASE_PATH` para reutilizar outro arquivo. A tabela usada pela aplicacao e
`dataset_lab`.

Na interface:

1. Abra **Dados**.
2. Selecione **Carregar dataset**.
3. Envie CSV ou XLSX.
4. Revise tipos, nulos e duplicados.
5. Confirme a importacao.

O DuckDB e a fonte correta para medias, medianas, minimos, maximos, filtros e
correlacoes. O RAG textual nao deve ser usado para calcular estatisticas.

## Modelo preditivo

Arquivos principais:

- `models/modelo_flaml.pkl`: pacote serializado do modelo;
- `models/metadata_modelo.json`: alvo, tarefa, features, medianas e metricas;
- `reports/metricas_modelo.csv`: metricas da ultima execucao;
- `reports/resultado_teste_modelo.csv`: valores reais e previstos.

Para usar um modelo existente, mantenha o `.pkl` e o JSON de metadados juntos.
Para treinar:

1. importe um dataset para `dataset_lab`;
2. abra **Modelos IA**;
3. escolha regressao ou classificacao;
4. selecione o alvo e o tempo maximo;
5. execute **Treinar modelo**.

O carregamento e feito por `ai/modeling/predict.py`. As entradas sao ordenadas
segundo `colunas_usadas`; valores ausentes usam `medianas_imputacao`.

## Agentes, chat e RAG

O chat classifica a pergunta antes de usar o LLM:

- `ai/intent_router.py` identifica intencao, confianca e entidades;
- `ai/numeric_query_engine.py` executa consultas com lista branca de colunas;
- perguntas numericas suportadas usam DuckDB e nao chamam LLM;
- perguntas sobre modelo usam metadados e metricas;
- perguntas documentais podem recuperar trechos do ChromaDB;
- perguntas gerais podem seguir para CrewAI e o provedor de LLM ativo.

O provedor do chat e independente do provedor de embeddings. Nesta versao, a
indexacao vetorial do RAG continua usando `OLLAMA_EMBED_MODEL` no Ollama.

O RAG é opcional. Quando `chromadb` não está instalado, a base vetorial não
existe ou não pode ser aberta, o Assistente informa que a recuperação de
documentos está indisponível e continua respondendo diretamente com Gemini,
OpenAI ou Ollama. O ambiente enxuto de Docker/Render não instala ChromaDB por
padrão. Para ativar RAG no deploy, adicione `chromadb` às dependências de
produção, disponibilize armazenamento persistente e indexe os documentos.

## Docker e deploy

O `.dockerignore` exclui `.env`, segredos do Streamlit, logs e caches. Passe as
variaveis no runtime ou pelo gerenciador de segredos da plataforma:

```powershell
docker build -t projeto-ia-analitica .
docker run --rm -p 8501:8501 `
  -e LLM_PROVIDER=gemini `
  -e GEMINI_MODEL=gemini-2.5-flash `
  -e GEMINI_API_KEY=$env:GEMINI_API_KEY `
  projeto-ia-analitica
```

O Dockerfile instala `requirements-deploy.txt`. Dependencias exclusivas de
testes, avaliacao de IA, Windows e desenvolvimento permanecem somente em
`requirements.txt`.

O ambiente de deploy usa a camada central LiteLLM diretamente para o chat.
CrewAI, ChromaDB e suas dependencias pesadas permanecem no ambiente completo de
desenvolvimento. Quando CrewAI esta instalado, a orquestracao local continua
disponivel; sem ele, o sistema usa o modo direto de producao.

Para um Ollama executado fora do container, `localhost` aponta para o proprio
container. Use um hostname acessivel pelo container, por exemplo
`http://host.docker.internal:11434` no Docker Desktop, ou execute os servicos na
mesma rede Docker.

### Deploy no Render

Crie um **Web Service** usando o Dockerfile do repositorio. O Render fornece a
variavel `PORT`, utilizada automaticamente pelo comando de inicializacao.

Configure no painel do servico:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=chave_configurada_como_secret
GEMINI_MODEL=gemini-2.5-flash
DATABASE_PATH=/tmp/lab_ia.duckdb
```

Para OpenAI, use alternativamente `LLM_PROVIDER=openai`, `OPENAI_API_KEY` e
`OPENAI_MODEL`. Nao coloque nenhuma chave no Dockerfile ou no repositorio. O filesystem do Render pode
ser efemero; arquivos enviados, banco DuckDB, traces e indices vetoriais podem
ser perdidos em reinicios ou novos deploys. Para persistencia, configure um
disco persistente e aponte os caminhos de dados para o ponto de montagem.

O container inicia somente o Streamlit. Se os recursos que dependem da FastAPI
forem usados, publique a API como um segundo servico e configure `API_BASE_URL`
com a URL desse servico.

A tolerancia padrao para filtros aproximados e `0.2`. Portanto, uma consulta por
proteina igual a 46% usa a faixa de 45,8% a 46,2%.

Documentos aceitos: PDF, DOCX, TXT e Markdown. Eles ficam em
`data/knowledge_base/` e os vetores em `data/vectorstore/`.

## Interpretacao das metricas

Para regressao:

- **MAE**: erro absoluto medio; menor e melhor;
- **RMSE**: penaliza erros grandes; menor e melhor;
- **R2**: proporcao da variacao explicada; maior e melhor, mas nao prova validade
  operacional.

Para classificacao:

- **Accuracy**: proporcao total de acertos;
- **Precision**: confiabilidade das previsoes positivas;
- **Recall**: cobertura dos casos positivos;
- **F1**: equilibrio entre precision e recall;
- **ROC-AUC**: separacao entre classes, quando aplicavel.

As metricas devem ser analisadas com tamanho da amostra, representatividade,
faixa de aplicacao, risco operacional e validacao externa.

## Testes

Testes minimos sem acesso real a provedores:

```powershell
python -m unittest discover -s tests/unit -v
```

Golden dataset dos agentes:

```powershell
python tests/integration/run_golden_tests.py --limite 3
```

Suite de qualidade de IA:

```powershell
python tests/ai/run_ai_tests.py --heuristic
```

Os testes com CrewAI/Ollama dependem de servicos externos locais e podem demorar.

## Logs e rastreabilidade

- `reports/traces_crewai_llm.jsonl`: execucoes dos agentes e provedor ativo;
- `reports/rag_numeric_router.jsonl`: classificacao e tempo das consultas numericas;
- `reports/predictions_history.jsonl`: historico de previsoes manuais;
- `reports/assistant_chat_history.jsonl`: historico do chat;
- `reports/logs/`: logs de inicializacao.

## Limitacoes conhecidas

- o roteador numerico cobre apenas padroes e variaveis explicitamente mapeados;
- respostas gerais dependem da disponibilidade e qualidade do provedor LLM ativo;
- o RAG recupera trechos, mas nao garante que o documento contenha a resposta;
- o modelo pode degradar fora das faixas observadas no treinamento;
- os dados atuais incluem registros simulados e nao representam validacao oficial;
- nao existe autenticacao, controle de perfis ou isolamento multiusuario;
- arquivos JSONL podem crescer e ainda nao possuem rotacao automatica;
- a interface principal ainda esta concentrada em `app/main.py`.

## Proximos passos

1. ampliar o roteador numerico para filtros e variaveis dinamicas;
2. versionar modelos e comparar experimentos;
3. separar as telas Streamlit em `app/pages/`;
4. adicionar autenticacao e autorizacao;
5. criar migracoes e esquema formal do DuckDB;
6. adicionar testes de contrato da API;
7. monitorar qualidade do RAG e respostas do LLM;
8. criar validacao externa do modelo com dados reais.

# Projeto IA Analitica Lab

Plataforma local para analise de dados laboratoriais, treinamento de modelos
preditivos, consultas numericas com DuckDB, agentes de IA com Ollama e recuperacao
de conhecimento por RAG.

## Objetivo

O projeto centraliza dados laboratoriais de soja, farelo e casca para:

- importar e explorar datasets;
- consultar dados com SQL;
- treinar modelos com FLAML AutoML;
- executar previsoes com o modelo salvo;
- responder perguntas numericas diretamente pelo DuckDB;
- responder perguntas conceituais com agentes CrewAI e Ollama;
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
| LLM local | Ollama |
| RAG e vetores | ChromaDB e embeddings Ollama |
| Graficos | Plotly e Matplotlib |
| Testes de IA | unittest, scripts golden dataset e DeepEval opcional |

## Arquitetura

```text
app/                    Interface Streamlit, configuracao e componentes
ai/
  intent_router.py      Classificacao de intencao e extracao de entidades
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
- Ollama para agentes e embeddings;
- modelo Ollama configurado no `.env`;
- ambiente virtual Python.

## Instalacao

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

O `.env.example` nao possui chaves reais. Ajuste o modelo Ollama e os caminhos
somente quando necessario.

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
- perguntas numericas suportadas usam DuckDB e nao chamam Ollama;
- perguntas sobre modelo usam metadados e metricas;
- perguntas documentais podem recuperar trechos do ChromaDB;
- perguntas gerais podem seguir para CrewAI e Ollama.

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

Testes minimos sem Ollama:

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

- `reports/traces_crewai_ollama.jsonl`: execucoes dos agentes;
- `reports/rag_numeric_router.jsonl`: classificacao e tempo das consultas numericas;
- `reports/predictions_history.jsonl`: historico de previsoes manuais;
- `reports/assistant_chat_history.jsonl`: historico do chat;
- `reports/logs/`: logs de inicializacao.

## Limitacoes conhecidas

- o roteador numerico cobre apenas padroes e variaveis explicitamente mapeados;
- respostas gerais dependem da disponibilidade e qualidade do modelo Ollama;
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

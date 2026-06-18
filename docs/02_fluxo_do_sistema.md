# Fluxo do sistema

## Inicializacao

1. `scripts/start_api.ps1` inicia `api.model_api:app` na porta 8000.
2. `scripts/start_app.ps1` inicia `app/main.py` na porta 8501.
3. `app/config.py` resolve caminhos a partir da raiz do projeto.
4. A interface verifica API, Ollama, DuckDB, modelo e RAG.

## Fluxo de dados

```text
CSV/XLSX
  -> validacao Streamlit
  -> pandas DataFrame
  -> DuckDB / tabela dataset_lab
  -> exploracao, SQL, treinamento e consultas numericas
```

Os dados brutos ficam em `data/raw/`. O banco local fica em
`data/lab_ia.duckdb`.

## Fluxo de treinamento

```text
dataset_lab
  -> selecao do alvo
  -> remocao de colunas inadequadas
  -> imputacao por mediana
  -> divisao treino/teste
  -> FLAML AutoML
  -> metricas e resultados
  -> modelo_flaml.pkl + metadata_modelo.json
```

## Fluxo de previsao

1. O JSON de metadados informa `colunas_usadas`.
2. O formulario coleta os valores.
3. Valores ausentes usam `medianas_imputacao`.
4. O DataFrame e ordenado como no treinamento.
5. O pacote joblib e carregado.
6. `predict()` retorna a previsao.
7. A interface registra o historico.

## Fluxo de perguntas

```text
Pergunta
  -> identificar_intencao()
      -> consulta_numerica: numeric_query_engine + DuckDB
      -> metricas_modelo: metadata_modelo.json
      -> colunas_treinamento: metadata_modelo.json
      -> limitacoes_modelo: resposta local segura
      -> fora_escopo: resposta segura
      -> interpretacao/predicao/geral: CrewAI + Ollama
```

Perguntas numericas suportadas nunca devem calcular estatisticas a partir de texto
RAG.

O roteamento registra pergunta, intencao, confianca, entidades, ferramenta, status
e tempo em `reports/intent_router.jsonl`.

## Fluxo RAG

```text
PDF/DOCX/TXT/MD
  -> extracao de texto
  -> divisao em chunks
  -> embedding Ollama
  -> ChromaDB
  -> busca por similaridade
  -> trechos e fontes
  -> agente/usuario
```

## Observabilidade

Os agentes registram pergunta, resposta, status, ferramentas, modelo e duracao em
`reports/traces_crewai_ollama.jsonl`. O roteador numerico registra tipo e tempo em
`reports/rag_numeric_router.jsonl`.

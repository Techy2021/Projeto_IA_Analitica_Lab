# Agentes de IA

## Responsabilidade

Os agentes traduzem perguntas em tarefas e interpretam resultados obtidos por
ferramentas. Eles nao devem inventar calculos, metricas ou resultados laboratoriais.

## Agentes atuais

- **Analista de qualidade**: interpreta dados fisico-quimicos e previsoes.
- **Especialista em modelo**: explica alvo, features, metricas e limitacoes.

Os papeis basicos ficam em `ai/prompts/laboratorio.py`. A orquestracao fica em
`ai/agentes/crewai_agents_lab.py`.

## Roteamento

Antes de criar agentes, o sistema classifica a pergunta:

- `consulta_numerica`;
- `metricas_modelo`;
- `colunas_treinamento`;
- `interpretacao_laboratorial`;
- `predicao_amostra`;
- `limitacoes_modelo`;
- `fora_escopo`;
- `geral_laboratorial`.

Perguntas numericas suportadas retornam diretamente do DuckDB. Nesse caso:

- `modelo_ollama` recebe `nao_utilizado`;
- `ferramenta_utilizada` inclui `numeric_query_engine` e `duckdb`;
- nenhum agente e instanciado.

## Ferramentas

Os agentes podem consultar:

- API de previsao;
- metadados do modelo;
- DuckDB por consultas SELECT;
- amostra media;
- base RAG.

## Uso do Ollama

O Ollama e usado para interpretacao textual e embeddings. As configuracoes principais
sao `OLLAMA_BASE_URL` e `OLLAMA_MODEL`.

## Respostas fora do escopo

Quando faltam dados, a resposta correta e informar a limitacao. O sistema nao deve
fabricar limites oficiais, valores ideais ou conclusoes de liberacao.

## Logs

Cada execucao registra:

- pergunta;
- resposta;
- status;
- agentes;
- ferramentas;
- modelo;
- tempo;
- erro, quando houver.

O arquivo e `reports/traces_crewai_ollama.jsonl`.

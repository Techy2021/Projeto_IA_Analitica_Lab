# Testes e validacao

## Camadas de teste

### Testes unitarios

Executam sem Ollama e verificam contratos basicos:

```powershell
python -m unittest discover -s tests/unit -v
```

Cobrem:

- existencia e carregamento do modelo;
- leitura dos metadados;
- leitura e exibicao textual das metricas;
- resposta rapida do chat;
- pergunta numerica;
- pergunta fora do escopo;
- registro de log.
- classificacao das intencoes principais;
- extracao de operacao, variavel, filtro e valor;
- media, maximo e contagem calculados no DuckDB.

### Testes de integracao

```powershell
python tests/integration/run_golden_tests.py --limite 3
```

Podem usar FastAPI, CrewAI e Ollama. Sao mais lentos e dependem do ambiente local.

### Testes de qualidade de IA

```powershell
python tests/ai/run_ai_tests.py --heuristic
```

Tambem existem modos com Ollama julgador ou DeepEval. Consulte `--help`.

## Interpretacao

Um teste tecnico aprovado confirma o contrato implementado, nao a validade cientifica
do laboratorio.

Para respostas de IA, observe:

- relevancia;
- fidelidade ao contexto;
- adequacao tecnica;
- ausencia de dados inventados;
- declaracao de limitacoes.

## Resultados

Arquivos de saida ficam em `reports/`, incluindo:

- `resultados_testes_ia.csv`;
- `resultados_golden_dataset.csv`;
- `resumo_falhas_testes_ia.csv`.

## Antes de publicar

1. execute os testes unitarios;
2. teste `/health`, `/metadata`, `/metricas` e `/perguntar`;
3. abra a interface e percorra as telas principais;
4. valide modelo e metadados juntos;
5. teste ao menos uma pergunta numerica e uma documental;
6. revise traces e erros.

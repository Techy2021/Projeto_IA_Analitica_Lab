# Guia para novos desenvolvedores

## Primeiro contato

Leia nesta ordem:

1. `README.md`;
2. `docs/01_visao_geral.md`;
3. `docs/02_fluxo_do_sistema.md`;
4. `app/config.py`;
5. `app/main.py`;
6. `api/model_api.py`.

## Preparar o ambiente

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

## Executar

```powershell
.\scripts\start_api.ps1
.\scripts\start_app.ps1
```

Use terminais separados.

## Onde alterar

- nova tela ou componente: `app/pages/` ou `app/components/`;
- novo endpoint: `api/model_api.py`;
- nova consulta: `database/consultas.py`;
- novo comportamento de agente: `ai/agentes/`;
- prompt reutilizavel: `ai/prompts/`;
- nova regra numerica: `ai/rag/router.py`;
- indexacao ou recuperacao: `ai/rag/`;
- treinamento ou inferencia: `ai/modeling/`;
- caminhos: `app/config.py`;
- teste rapido: `tests/unit/`.

## Convencoes

- use caminhos derivados de `BASE_DIR`;
- nao use strings de caminho absolutas;
- mantenha calculos numericos no DuckDB;
- nao calcule estatisticas a partir de texto RAG;
- registre limitacoes quando faltarem dados;
- preserve o contrato de `metadata_modelo.json`;
- nao grave chaves em arquivos versionados.

## Checklist de mudanca

1. localizar todos os usos do arquivo ou funcao;
2. implementar a menor alteracao necessaria;
3. adicionar ou ajustar teste;
4. executar `py_compile`;
5. executar testes unitarios;
6. validar Streamlit e FastAPI;
7. revisar logs;
8. atualizar documentacao.

## Diagnostico rapido

- API offline: confira porta 8000 e `.env`;
- interface offline: confira porta 8501;
- Ollama offline: execute `ollama serve`;
- modelo ausente: verifique `models/modelo_flaml.pkl`;
- metadados ausentes: verifique `models/metadata_modelo.json`;
- RAG vazio: indexe documentos de `data/knowledge_base/`;
- banco vazio: importe dataset para `dataset_lab`.

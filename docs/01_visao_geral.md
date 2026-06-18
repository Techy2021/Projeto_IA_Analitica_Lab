# Visao geral

## O que e o projeto

O Projeto IA Analitica Lab e uma plataforma local para apoiar a analise de dados
laboratoriais relacionados ao processamento de soja. Ele combina banco analitico,
modelo preditivo, agentes de IA e busca documental.

## Problemas atendidos

- centralizar dados de soja, farelo e casca;
- calcular estatisticas com rastreabilidade;
- treinar e reutilizar modelos preditivos;
- responder perguntas tecnicas em linguagem natural;
- consultar documentos internos;
- registrar execucoes para auditoria e testes.

## Componentes

1. **Streamlit** apresenta dashboards, formularios, chat e observabilidade.
2. **FastAPI** disponibiliza previsao, metadados, consultas e perguntas roteadas.
3. **DuckDB** armazena a tabela `dataset_lab` e executa calculos numericos.
4. **FLAML** seleciona um estimador durante o treinamento AutoML.
5. **Ollama** executa o LLM local e gera embeddings.
6. **CrewAI** coordena agentes especializados.
7. **ChromaDB** armazena vetores dos documentos da base de conhecimento.

## Principio de arquitetura

Cada fonte tem uma responsabilidade:

- DuckDB calcula numeros;
- o modelo `.pkl` gera previsoes;
- `metadata_modelo.json` descreve o contrato do modelo;
- ChromaDB recupera trechos documentais;
- Ollama interpreta linguagem e contexto;
- logs JSONL registram o que aconteceu.

## Limite de responsabilidade

O sistema apoia analistas, mas nao deve liberar lotes automaticamente nem substituir
procedimentos oficiais, calibracao, rastreabilidade e revisao tecnica.

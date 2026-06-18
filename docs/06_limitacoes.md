# Limitacoes

## Dados

- Os dados podem incluir registros simulados.
- A representatividade de outras safras, plantas ou equipamentos nao e garantida.
- Alteracoes de unidade ou metodo analitico podem invalidar comparacoes.

## Modelo

- O modelo aprende associacoes, nao causalidade.
- Desempenho interno nao garante desempenho futuro.
- Valores fora das faixas de treinamento podem gerar previsoes inadequadas.
- O artefato atual e sobrescrito durante novo treinamento.

## Perguntas numericas

- O roteador depende de termos e variaveis mapeadas.
- A primeira versao usa faixas predefinidas para proteina proxima de 46%.
- Perguntas numericas ainda nao mapeadas retornam limitacao.

## Agentes e LLM

- Ollama pode estar indisponivel, lento ou sem o modelo configurado.
- O LLM pode interpretar incorretamente textos ambiguos.
- CrewAI adiciona latencia e dependencias.
- Nenhuma resposta deve ser tratada como laudo.

## RAG

- Recuperacao por similaridade nao garante resposta completa.
- PDFs digitalizados sem OCR podem produzir pouco texto.
- Documentos desatualizados continuam recuperaveis ate serem removidos.
- A indexacao depende do modelo de embeddings.

## Operacao

- Nao ha autenticacao ou gestao de usuarios.
- Nao ha criptografia especifica dos artefatos locais.
- JSONL nao possui rotacao automatica.
- DuckDB local nao foi projetado para escrita concorrente intensa.
- A aplicacao ainda nao possui implantacao containerizada oficial.

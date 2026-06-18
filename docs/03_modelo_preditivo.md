# Modelo preditivo

## Objetivo atual

O modelo salvo atualmente e uma regressao cujo alvo e
`farelo_proteina_pct`. O estimador selecionado na ultima execucao foi
`extra_tree`.

Consulte sempre `models/metadata_modelo.json`, pois o alvo e o estimador podem
mudar apos novo treinamento.

## Artefatos

- `models/modelo_flaml.pkl`: pacote serializado com o modelo;
- `models/metadata_modelo.json`: contrato e contexto do treinamento;
- `reports/metricas_modelo.csv`: metricas;
- `reports/resultado_teste_modelo.csv`: comparacao real versus previsto.

## Conteudo dos metadados

Campos importantes:

- `alvo`: variavel prevista;
- `tipo_problema`: regression ou classification;
- `melhor_estimador`: algoritmo escolhido;
- `colunas_usadas`: ordem das entradas;
- `medianas_imputacao`: valores para entradas ausentes;
- `metricas`: desempenho no conjunto de teste;
- `data_treinamento`: data da geracao;
- `linhas_treino` e `linhas_teste`: volume utilizado.

## Carregamento

`ai/modeling/predict.py`:

1. verifica se `.pkl` e JSON existem;
2. le e valida o JSON;
3. carrega o pacote com `joblib`;
4. identifica a chave do modelo;
5. monta a entrada na ordem correta;
6. chama `predict()`.

Nunca envie ao modelo um DataFrame com ordem ou nomes de colunas diferentes de
`colunas_usadas`.

## Metricas atuais

Na ultima execucao:

- MAE aproximado: 0,278;
- RMSE aproximado: 0,341;
- R2 aproximado: 0,677.

Esses valores descrevem o teste interno. Eles nao representam garantia de desempenho
em novos lotes ou em outro laboratorio.

## Retreinamento

O treinamento sobrescreve os artefatos atuais. Antes de retreinar em ambiente
controlado, copie os artefatos ou implemente versionamento.

## Cuidados

- verificar vazamento de alvo;
- validar unidades e metodos analiticos;
- monitorar valores fora da faixa de treinamento;
- comparar desempenho por periodo, turno, produto e equipamento;
- validar com dados reais independentes.

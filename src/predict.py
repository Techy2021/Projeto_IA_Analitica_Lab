import json
from typing import Any

import joblib
import pandas as pd

from src.config import METADATA_MODEL_PATH, MODEL_PATH


def modelo_treinado_existe() -> bool:
    return MODEL_PATH.exists() and METADATA_MODEL_PATH.exists()


def carregar_metadata_modelo() -> dict:
    if not METADATA_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Metadados do modelo nao encontrados. Treine um modelo antes de gerar previsoes."
        )

    with open(METADATA_MODEL_PATH, "r", encoding="utf-8") as arquivo:
        metadata = json.load(arquivo)

    if not isinstance(metadata, dict):
        raise ValueError("O arquivo de metadados do modelo esta em formato invalido.")

    return metadata


def carregar_pacote_modelo() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Nenhum modelo treinado encontrado. Treine um modelo antes de gerar previsoes."
        )

    return joblib.load(MODEL_PATH)


def carregar_modelo_salvo() -> Any:
    pacote = carregar_pacote_modelo()

    if isinstance(pacote, dict):
        modelo = pacote.get("modelo") or pacote.get("automl") or pacote.get("model")
        if modelo is None:
            raise ValueError("O pacote salvo nao contem uma chave de modelo reconhecida.")
        return modelo

    if hasattr(pacote, "predict"):
        return pacote

    raise ValueError(
        "O arquivo de modelo salvo nao possui uma estrutura reconhecida para previsao."
    )


def obter_colunas_usadas(metadata: dict) -> list[str]:
    colunas = metadata.get("colunas_usadas")

    if not colunas and isinstance(metadata.get("features"), list):
        colunas = metadata["features"]

    if not colunas or not isinstance(colunas, list):
        raise ValueError("Nao foi possivel identificar as colunas usadas no treinamento.")

    return [str(coluna) for coluna in colunas]


def obter_medianas_imputacao(metadata: dict) -> dict:
    medianas = metadata.get("medianas_imputacao")
    if isinstance(medianas, dict):
        return medianas
    return {}


def obter_info_modelo(metadata: dict) -> dict:
    return {
        "alvo": metadata.get("alvo", "Nao informado"),
        "tipo_problema": metadata.get(
            "tipo_problema",
            metadata.get("tarefa", "Nao informado"),
        ),
        "melhor_estimador": metadata.get("melhor_estimador", "Nao informado"),
        "colunas_usadas": obter_colunas_usadas(metadata),
    }


def montar_dataframe_previsao(
    valores_digitados: dict,
    colunas_usadas: list[str],
    medianas_imputacao: dict | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    medianas_imputacao = medianas_imputacao or {}
    valores_linha = {}
    colunas_preenchidas = []
    colunas_extras = [
        coluna for coluna in valores_digitados.keys() if coluna not in colunas_usadas
    ]

    for coluna in colunas_usadas:
        if coluna not in valores_digitados or valores_digitados[coluna] in ("", None):
            valor = medianas_imputacao.get(coluna, 0)
            colunas_preenchidas.append(coluna)
        else:
            valor = valores_digitados[coluna]

        try:
            valores_linha[coluna] = float(valor)
        except (TypeError, ValueError) as erro:
            raise ValueError(f"O campo '{coluna}' precisa receber um valor numerico.") from erro

    dataframe = pd.DataFrame([valores_linha], columns=colunas_usadas)
    return dataframe, colunas_preenchidas, colunas_extras


def gerar_previsao(valores_digitados: dict) -> Any:
    resultado = gerar_previsao_detalhada(valores_digitados)
    return resultado["previsao"]


def gerar_previsao_detalhada(valores_digitados: dict) -> dict:
    metadata = carregar_metadata_modelo()
    modelo = carregar_modelo_salvo()
    colunas_usadas = obter_colunas_usadas(metadata)
    entrada, colunas_preenchidas, colunas_extras = montar_dataframe_previsao(
        valores_digitados,
        colunas_usadas,
        obter_medianas_imputacao(metadata),
    )

    previsao = modelo.predict(entrada)

    if hasattr(previsao, "tolist"):
        previsao = previsao.tolist()

    if isinstance(previsao, list):
        if not previsao:
            raise ValueError("O modelo nao retornou nenhuma previsao.")
        previsao = previsao[0]

    info = obter_info_modelo(metadata)
    return {
        "previsao": previsao,
        "alvo": info["alvo"],
        "tarefa": info["tipo_problema"],
        "colunas_usadas": colunas_usadas,
        "colunas_preenchidas": colunas_preenchidas,
        "colunas_extras_ignoradas": colunas_extras,
    }

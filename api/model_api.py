from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ai.modeling.predict import (
    carregar_metadata_modelo,
    gerar_previsao_detalhada,
    obter_colunas_usadas,
)
from ai.intent_router import identificar_intencao
from ai.numeric_query_engine import executar_consulta_numerica, formatar_resposta_consulta
from database.consultas import consultar_sql


app = FastAPI(title="API IA Analitica Laboratorial")


class PrevisaoRequest(BaseModel):
    dados: dict[str, Any] = Field(default_factory=dict)


class ConsultaRequest(BaseModel):
    query: str


class PerguntaRequest(BaseModel):
    pergunta: str


COMANDOS_BLOQUEADOS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "REPLACE",
    "MERGE",
    "COPY",
    "ATTACH",
    "DETACH",
}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "mensagem": "API IA Analítica Laboratorial ativa",
    }


@app.get("/metadata")
def metadata():
    """Expoe somente os campos de metadados necessarios aos consumidores da API."""
    try:
        dados = carregar_metadata_modelo()
        return {
            "alvo": dados.get("alvo"),
            "tarefa": dados.get("tipo_problema", dados.get("tarefa")),
            "melhor_estimador": dados.get("melhor_estimador"),
            "colunas_usadas": dados.get("colunas_usadas", []),
            "metricas": dados.get("metricas", {}),
        }
    except Exception as erro:
        raise HTTPException(status_code=404, detail=str(erro)) from erro


@app.get("/metricas")
def metricas():
    """Retorna o metadata completo, incluindo metricas da ultima versao treinada."""
    try:
        return carregar_metadata_modelo()
    except Exception as erro:
        raise HTTPException(status_code=404, detail=str(erro)) from erro


@app.post("/prever")
def prever(payload: PrevisaoRequest):
    try:
        resultado = gerar_previsao_detalhada(payload.dados)
        resposta = {
            "previsao": resultado["previsao"],
            "alvo": resultado["alvo"],
            "tarefa": resultado["tarefa"],
            "colunas_preenchidas": resultado["colunas_preenchidas"],
            "colunas_extras_ignoradas": resultado["colunas_extras_ignoradas"],
        }
        if resultado["tarefa"] == "classification":
            resposta["classe_prevista"] = str(resultado["previsao"])
        return resposta
    except Exception as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro


@app.post("/consultar")
def consultar(payload: ConsultaRequest):
    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Consulta vazia.")

    query_upper = query.upper()
    primeira_palavra = query_upper.split()[0]
    if primeira_palavra != "SELECT":
        raise HTTPException(status_code=400, detail="Apenas consultas SELECT sao permitidas.")

    tokens = set(query_upper.replace(";", " ").split())
    if tokens.intersection(COMANDOS_BLOQUEADOS):
        raise HTTPException(
            status_code=400,
            detail="Consulta bloqueada por conter comando destrutivo.",
        )

    try:
        df = consultar_sql(query)
        return {
            "linhas": int(len(df)),
            "registros": df.where(pd.notna(df), None).to_dict(orient="records"),
        }
    except Exception as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro


@app.post("/perguntar")
def perguntar(payload: PerguntaRequest):
    """Roteia perguntas numericas para DuckDB sem acionar um LLM."""
    pergunta_usuario = payload.pergunta.strip()
    if not pergunta_usuario:
        raise HTTPException(status_code=400, detail="Pergunta vazia.")

    roteamento = identificar_intencao(pergunta_usuario)
    tipo = roteamento["intencao"]
    if tipo == "consulta_numerica":
        try:
            resultado = executar_consulta_numerica(roteamento)
            return {
                "tipo": tipo,
                "roteamento": roteamento,
                "fonte": "duckdb",
                "chamou_ollama": False,
                "chamou_llm": False,
                "resultado": resultado,
                "resposta": formatar_resposta_consulta(resultado),
            }
        except Exception as erro:
            raise HTTPException(status_code=400, detail=str(erro)) from erro

    from ai.agentes.crewai_agents_lab import executar_crew_lab

    resultado = executar_crew_lab(pergunta_usuario)
    return {
        "tipo": tipo,
        "roteamento": roteamento,
        "fonte": "roteador",
        "chamou_ollama": resultado.get("llm_provider") == "ollama",
        "chamou_llm": resultado.get("llm_provider") not in {None, "nao_utilizado"},
        "provedor_llm": resultado.get("llm_provider"),
        "modelo_llm": resultado.get("llm_model"),
        "resultado": resultado,
        "resposta": resultado.get("resposta", ""),
    }

    return {
        "tipo": tipo,
        "fonte": "roteador",
        "chamou_ollama": False,
        "resposta": (
            "Pergunta classificada como "
            f"{tipo}. Nesta versão, apenas perguntas numéricas são respondidas "
            "diretamente pelo DuckDB neste endpoint."
        ),
    }


@app.get("/amostra")
def amostra():
    try:
        metadata_modelo = carregar_metadata_modelo()
        colunas_usadas = obter_colunas_usadas(metadata_modelo)
        colunas_sql = ", ".join(
            [f'AVG("{coluna}") AS "{coluna}"' for coluna in colunas_usadas]
        )
        df = consultar_sql(f"SELECT {colunas_sql} FROM dataset_lab")
        if df.empty:
            raise ValueError("Nao foi possivel gerar amostra media.")
        return {
            "dados": df.iloc[0].where(pd.notna(df.iloc[0]), 0).to_dict(),
            "colunas_usadas": colunas_usadas,
        }
    except Exception as erro:
        raise HTTPException(status_code=400, detail=str(erro)) from erro

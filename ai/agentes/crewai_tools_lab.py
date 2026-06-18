import json
import os
import re
import traceback
from typing import Any, Type

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, Field


API_BASE_URL_PADRAO = "http://localhost:8000"
MENSAGEM_API_INDISPONIVEL = (
    "API local não encontrada. Inicie com: "
    "python -m uvicorn api.model_api:app --reload --port 8000"
)
COMANDOS_BLOQUEADOS = {
    "DROP",
    "DELETE",
    "UPDATE",
    "INSERT",
    "ALTER",
    "CREATE",
    "TRUNCATE",
}


try:
    from crewai.tools import BaseTool

    CREWAI_TOOLS_DISPONIVEL = True
    CREWAI_TOOLS_IMPORT_ERROR = None
    CREWAI_TOOLS_IMPORT_TRACEBACK = None
except ModuleNotFoundError as erro:
    BaseTool = object
    CREWAI_TOOLS_DISPONIVEL = False
    CREWAI_TOOLS_IMPORT_ERROR = (
        f"Pacote não encontrado: {erro.name}. Execute: "
        "python -m pip install -r requirements.txt"
    )
    CREWAI_TOOLS_IMPORT_TRACEBACK = traceback.format_exc()
except Exception as erro:
    BaseTool = object
    CREWAI_TOOLS_DISPONIVEL = False
    CREWAI_TOOLS_IMPORT_ERROR = (
        f"Erro ao importar ferramentas CrewAI: {type(erro).__name__}: {erro}"
    )
    CREWAI_TOOLS_IMPORT_TRACEBACK = traceback.format_exc()


class PreverFareloSojaInput(BaseModel):
    umidade_pct: float = Field(..., description="Percentual de umidade.")
    proteina_pct: float = Field(..., description="Percentual de proteína.")
    extrato_etereo_pct: float = Field(..., description="Percentual de extrato etéreo.")
    fibras_pct: float = Field(..., description="Percentual de fibras.")
    materia_mineral_pct: float = Field(..., description="Percentual de matéria mineral.")
    urease_uph: float = Field(..., description="Valor de urease em UPH.")
    solubilidade_pct: float = Field(..., description="Percentual de solubilidade.")


class ConsultaDadosLaboratorioInput(BaseModel):
    query: str = Field(..., description="Consulta SQL SELECT para executar no DuckDB.")


def _resposta_api(requisicao):
    try:
        resposta = requisicao()
        resposta.raise_for_status()
        return resposta.json()
    except requests.exceptions.ConnectionError:
        return {"erro": MENSAGEM_API_INDISPONIVEL}
    except requests.exceptions.Timeout:
        return {"erro": "Tempo excedido ao chamar a API local configurada em API_BASE_URL."}
    except requests.exceptions.HTTPError as erro:
        try:
            detalhe = erro.response.json().get("detail", str(erro))
        except Exception:
            detalhe = str(erro)
        return {"erro": detalhe}
    except Exception as erro:
        return {"erro": f"Erro ao chamar a API local: {erro}"}


def _formatar(resultado: Any) -> str:
    return json.dumps(resultado, ensure_ascii=False, indent=2)


def _normalizar_argumentos(args: tuple, kwargs: dict) -> dict:
    if args and isinstance(args[0], dict):
        dados = dict(args[0])
        dados.update(kwargs)
        return dados
    return dict(kwargs)


def obter_api_base_url() -> str:
    load_dotenv()
    return os.getenv("API_BASE_URL", API_BASE_URL_PADRAO).rstrip("/")


def prever_farelo_soja_func(*args, **kwargs) -> str:
    dados = _normalizar_argumentos(args, kwargs)
    api_base_url = obter_api_base_url()
    resultado = _resposta_api(
        lambda: requests.post(
            f"{api_base_url}/prever",
            json={"dados": dados},
            timeout=15,
        )
    )
    return _formatar(resultado)


def _validar_select_seguro(query: str) -> str | None:
    query_limpa = (query or "").strip()
    if not query_limpa:
        return "Consulta vazia. Informe uma consulta SELECT."
    if not query_limpa.upper().startswith("SELECT"):
        return "Consulta bloqueada. Apenas comandos SELECT são permitidos."

    tokens = set(re.findall(r"\b[A-Z_]+\b", query_limpa.upper()))
    comandos_encontrados = sorted(tokens.intersection(COMANDOS_BLOQUEADOS))
    if comandos_encontrados:
        return (
            "Consulta bloqueada por conter comando destrutivo: "
            + ", ".join(comandos_encontrados)
        )
    return None


def consultar_dados_laboratorio_func(query: str) -> str:
    erro_validacao = _validar_select_seguro(query)
    if erro_validacao:
        return _formatar({"erro": erro_validacao})

    api_base_url = obter_api_base_url()
    resultado = _resposta_api(
        lambda: requests.post(
            f"{api_base_url}/consultar",
            json={"query": query},
            timeout=20,
        )
    )
    return _formatar(resultado)


def obter_metadata_modelo_func() -> str:
    api_base_url = obter_api_base_url()
    resultado = _resposta_api(
        lambda: requests.get(f"{api_base_url}/metadata", timeout=15)
    )
    return _formatar(resultado)


def obter_amostra_media_func() -> str:
    api_base_url = obter_api_base_url()
    resultado = _resposta_api(
        lambda: requests.get(f"{api_base_url}/amostra", timeout=15)
    )
    return _formatar(resultado)


if CREWAI_TOOLS_DISPONIVEL:

    class PreverFareloSojaTool(BaseTool):
        name: str = "prever_farelo_soja"
        description: str = (
            "Usa o modelo FLAML treinado para prever a qualidade de uma amostra "
            "de farelo de soja."
        )
        args_schema: Type[BaseModel] = PreverFareloSojaInput

        def _run(self, **kwargs: Any) -> str:
            return prever_farelo_soja_func(kwargs)


    class ConsultarDadosLaboratorioTool(BaseTool):
        name: str = "consultar_dados_laboratorio"
        description: str = "Executa uma consulta SQL SELECT no DuckDB por meio da API."
        args_schema: Type[BaseModel] = ConsultaDadosLaboratorioInput

        def _run(self, query: str) -> str:
            return consultar_dados_laboratorio_func(query)


    class ObterMetadataModeloTool(BaseTool):
        name: str = "obter_metadata_modelo"
        description: str = "Consulta os metadados do modelo FLAML treinado."

        def _run(self) -> str:
            return obter_metadata_modelo_func()


    class ObterAmostraMediaTool(BaseTool):
        name: str = "obter_amostra_media"
        description: str = "Obtém uma amostra média das variáveis usadas pelo modelo."

        def _run(self) -> str:
            return obter_amostra_media_func()


    prever_farelo_soja = PreverFareloSojaTool()
    consultar_dados_laboratorio = ConsultarDadosLaboratorioTool()
    obter_metadata_modelo = ObterMetadataModeloTool()
    obter_amostra_media = ObterAmostraMediaTool()
else:
    prever_farelo_soja = None
    consultar_dados_laboratorio = None
    obter_metadata_modelo = None
    obter_amostra_media = None

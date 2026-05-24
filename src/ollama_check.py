import os
from typing import Any

import requests
from dotenv import load_dotenv


OLLAMA_BASE_URL_PADRAO = "http://localhost:11434"
OLLAMA_MODEL_PADRAO = "llama3.2:3b"
MENSAGEM_OLLAMA_INDISPONIVEL = (
    "Ollama não encontrado. Abra o Ollama ou execute ollama serve."
)
MENSAGEM_MODELO_INDISPONIVEL = (
    "Modelo não encontrado. Execute: ollama pull llama3.2:3b"
)


def obter_config_ollama() -> tuple[str, str]:
    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL_PADRAO).rstrip("/")
    modelo = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_PADRAO)
    return base_url, modelo


def _extrair_modelos(payload: dict[str, Any]) -> list[str]:
    modelos = []
    for item in payload.get("models", []):
        nome = item.get("name") or item.get("model")
        if nome:
            modelos.append(str(nome))
    return modelos


def verificar_ollama() -> dict:
    base_url, _ = obter_config_ollama()
    try:
        resposta = requests.get(f"{base_url}/api/tags", timeout=5)
        resposta.raise_for_status()
        modelos = _extrair_modelos(resposta.json())
        return {
            "status": "ok",
            "base_url": base_url,
            "modelos": modelos,
            "mensagem": "Ollama disponível.",
        }
    except requests.exceptions.RequestException:
        return {
            "status": "erro",
            "base_url": base_url,
            "modelos": [],
            "mensagem": MENSAGEM_OLLAMA_INDISPONIVEL,
        }
    except Exception as erro:
        return {
            "status": "erro",
            "base_url": base_url,
            "modelos": [],
            "mensagem": f"Erro ao verificar Ollama: {erro}",
        }


def modelo_ollama_disponivel(nome_modelo: str) -> dict:
    status_ollama = verificar_ollama()
    if status_ollama["status"] != "ok":
        return status_ollama

    modelos = status_ollama.get("modelos", [])
    disponivel = nome_modelo in modelos
    return {
        "status": "ok" if disponivel else "erro",
        "modelo": nome_modelo,
        "modelos": modelos,
        "mensagem": (
            f"Modelo {nome_modelo} disponível."
            if disponivel
            else MENSAGEM_MODELO_INDISPONIVEL
        ),
    }

from pathlib import Path
import sys

import pandas as pd
import requests
from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from ai.agentes.crewai_agents_lab import executar_crew_lab
from ai.agentes.crewai_tools_lab import obter_api_base_url
from ai.agentes.ollama_check import modelo_ollama_disponivel, obter_config_ollama, verificar_ollama


PERGUNTAS = [
    (
        "Faça uma previsão para umidade 12.5, proteína 46.2, extrato etéreo 1.8, "
        "fibras 4.7, matéria mineral 4.5, urease 0.12 e solubilidade 82."
    ),
    "Quantas amostras existem por classe de qualidade?",
    "Quais são as médias por classe?",
    "Explique as métricas do modelo atual.",
    "Quais colunas foram usadas no treinamento?",
]


def verificar_api_fastapi() -> dict:
    api_base_url = obter_api_base_url()
    try:
        resposta = requests.get(f"{api_base_url}/health", timeout=5)
        resposta.raise_for_status()
        return {"status": "ok", "mensagem": "API FastAPI disponível."}
    except requests.exceptions.RequestException:
        return {
            "status": "erro",
            "mensagem": (
                "API local não encontrada. Inicie com: "
                "python -m uvicorn api.model_api:app --reload --port 8000"
            ),
        }


def main():
    load_dotenv()
    _, modelo = obter_config_ollama()
    status_ollama = verificar_ollama()
    status_modelo = modelo_ollama_disponivel(modelo)
    status_api = verificar_api_fastapi()

    print(status_ollama.get("mensagem"))
    print(status_modelo.get("mensagem"))
    print(status_api.get("mensagem"))

    resultados = []
    for pergunta in PERGUNTAS:
        resultado = executar_crew_lab(pergunta)
        resultados.append(
            {
                "pergunta": pergunta,
                "status": resultado.get("status"),
                "modelo_ollama": resultado.get("modelo_ollama"),
                "resposta": resultado.get("resposta"),
                "agentes": ", ".join(resultado.get("agentes", [])),
            }
        )

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    caminho_saida = reports_dir / "resultados_crewai_ollama.csv"
    pd.DataFrame(resultados).to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    print(f"Resultados salvos em: {caminho_saida}")


if __name__ == "__main__":
    main()

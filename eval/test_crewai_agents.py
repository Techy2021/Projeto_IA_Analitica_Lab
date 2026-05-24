from pathlib import Path
import sys

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.crewai_agents_lab import executar_crew_lab


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


def main():
    resultados = []
    for pergunta in PERGUNTAS:
        resultado = executar_crew_lab(pergunta)
        resultados.append(
            {
                "pergunta": pergunta,
                "status": resultado.get("status"),
                "resposta": resultado.get("resposta"),
                "agentes": ", ".join(resultado.get("agentes", [])),
            }
        )

    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)
    caminho_saida = reports_dir / "resultados_crewai_agents.csv"
    pd.DataFrame(resultados).to_csv(caminho_saida, index=False, encoding="utf-8-sig")
    print(f"Resultados salvos em: {caminho_saida}")


if __name__ == "__main__":
    main()

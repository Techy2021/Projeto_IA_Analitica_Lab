import csv
import json
import sys
import time
import argparse
from pathlib import Path


PROJETO_DIR = Path(__file__).resolve().parents[2]
if str(PROJETO_DIR) not in sys.path:
    sys.path.insert(0, str(PROJETO_DIR))

from ai.agentes.crewai_agents_lab import executar_crew_lab
from app.config import REPORTS_DIR, criar_pastas


GOLDEN_PATH = PROJETO_DIR / "tests" / "integration" / "golden_dataset.json"
RESULTADOS_PATH = REPORTS_DIR / "resultados_golden_dataset.csv"


def _texto_lista(valor) -> str:
    if isinstance(valor, list):
        return ", ".join(str(item) for item in valor)
    return str(valor or "")


def carregar_golden_dataset() -> list[dict]:
    with GOLDEN_PATH.open("r", encoding="utf-8") as arquivo:
        dados = json.load(arquivo)
    if not isinstance(dados, list):
        raise ValueError("tests/integration/golden_dataset.json deve conter uma lista.")
    return dados


def salvar_resultados(resultados: list[dict]) -> None:
    """Persiste o golden dataset em CSV para comparacao entre execucoes."""
    RESULTADOS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTADOS_PATH.open("w", encoding="utf-8-sig", newline="") as arquivo:
        campos = [
            "id",
            "pergunta",
            "tipo",
            "resposta",
            "status",
            "agente",
            "tempo_execucao_ms",
            "erro",
        ]
        escritor = csv.DictWriter(arquivo, fieldnames=campos)
        escritor.writeheader()
        escritor.writerows(resultados)


def main() -> int:
    """Executa perguntas de referencia e resume status, latencia e ferramentas."""
    parser = argparse.ArgumentParser(
        description="Executa o golden dataset dos agentes CrewAI + Ollama."
    )
    parser.add_argument(
        "--limite",
        type=int,
        default=None,
        help="Executa apenas as primeiras N perguntas, útil para smoke test.",
    )
    args = parser.parse_args()

    criar_pastas()
    perguntas = carregar_golden_dataset()
    if args.limite:
        perguntas = perguntas[: args.limite]

    resultados = []
    inicio_geral = time.perf_counter()

    for indice, item in enumerate(perguntas, start=1):
        pergunta_id = item.get("id", f"G{indice:03d}")
        pergunta = item.get("pergunta", "")
        tipo = item.get("tipo", "indefinido")

        print(f"[{indice}/{len(perguntas)}] {pergunta_id} - {tipo}")
        resultado = executar_crew_lab(pergunta)

        resultados.append(
            {
                "id": pergunta_id,
                "pergunta": pergunta,
                "tipo": tipo,
                "resposta": resultado.get("resposta", ""),
                "status": resultado.get("status", "erro"),
                "agente": _texto_lista(resultado.get("agentes", [])),
                "tempo_execucao_ms": resultado.get("tempo_execucao_ms", ""),
                "erro": resultado.get("resposta", "")
                if resultado.get("status") == "erro"
                else "",
            }
        )
        salvar_resultados(resultados)
        print(f"Resultado parcial salvo em: {RESULTADOS_PATH}")

    total = len(resultados)
    sucessos = sum(1 for item in resultados if item["status"] == "ok")
    erros = total - sucessos
    tempos = [
        float(item["tempo_execucao_ms"])
        for item in resultados
        if item["tempo_execucao_ms"] not in ("", None)
    ]
    tempo_medio = sum(tempos) / len(tempos) if tempos else 0.0
    tempo_total = time.perf_counter() - inicio_geral

    print("\nResumo golden dataset")
    print(f"Total de perguntas: {total}")
    print(f"Perguntas respondidas com sucesso: {sucessos}")
    print(f"Perguntas com erro: {erros}")
    print(f"Tempo médio de resposta: {tempo_medio:.0f} ms")
    print(f"Tempo total: {tempo_total:.1f} s")
    print(f"CSV gerado: {RESULTADOS_PATH}")

    return 0 if erros == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

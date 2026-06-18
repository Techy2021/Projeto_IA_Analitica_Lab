import csv
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTADOS_PATH = PROJECT_ROOT / "reports" / "resultados_testes_ia.csv"
RESUMO_PATH = PROJECT_ROOT / "reports" / "resumo_falhas_testes_ia.csv"


def main() -> None:
    resultados = ler_resultados(RESULTADOS_PATH)
    reprovados = [
        item
        for item in resultados
        if item.get("status_final", "").strip().upper() == "REPROVADO"
    ]

    imprimir_falhas(reprovados)
    imprimir_resumo(reprovados)
    salvar_resumo_falhas(reprovados, RESUMO_PATH)


def ler_resultados(caminho: Path) -> list[dict[str, Any]]:
    if not caminho.exists():
        print(f"Arquivo nao encontrado: {caminho}")
        sys.exit(1)

    with caminho.open("r", encoding="utf-8-sig", newline="") as arquivo:
        return list(csv.DictReader(arquivo))


def imprimir_falhas(reprovados: list[dict[str, Any]]) -> None:
    if not reprovados:
        print("Nenhum teste reprovado encontrado.")
        return

    print("\nTestes reprovados")
    print("=" * 80)

    for item in reprovados:
        print(f"\nID: {item.get('id', '')}")
        print(f"Categoria: {item.get('categoria', '')}")
        print(f"Pergunta: {item.get('pergunta', '')}")
        print(f"Resposta esperada: {item.get('resposta_esperada', '')}")
        print(f"Resposta obtida: {item.get('resposta_obtida', '')}")
        print(
            "Scores: "
            f"relevancia={item.get('score_relevancia', '')}, "
            f"fidelidade={item.get('score_fidelidade', '')}, "
            f"adequacao_tecnica={item.get('score_adequacao_tecnica', '')}"
        )
        print(f"Justificativa: {item.get('justificativa', '')}")
        print("-" * 80)


def imprimir_resumo(reprovados: list[dict[str, Any]]) -> None:
    total = len(reprovados)
    falhas_por_categoria = Counter(item.get("categoria", "") for item in reprovados)

    print("\nResumo das falhas")
    print(f"Total de reprovados: {total}")
    print("Quantidade de falhas por categoria:")
    if falhas_por_categoria:
        for categoria, quantidade in falhas_por_categoria.most_common():
            print(f"- {categoria}: {quantidade}")
    else:
        print("- nenhuma")

    print(f"Menor score de relevancia: {menor_score(reprovados, 'score_relevancia')}")
    print(f"Menor score de fidelidade: {menor_score(reprovados, 'score_fidelidade')}")
    print(
        "Menor score de adequacao tecnica: "
        f"{menor_score(reprovados, 'score_adequacao_tecnica')}"
    )


def salvar_resumo_falhas(reprovados: list[dict[str, Any]], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "id",
        "categoria",
        "pergunta",
        "score_relevancia",
        "score_fidelidade",
        "score_adequacao_tecnica",
        "justificativa",
    ]

    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        for item in reprovados:
            writer.writerow({campo: item.get(campo, "") for campo in campos})

    print(f"\nResumo salvo em: {caminho}")


def menor_score(registros: list[dict[str, Any]], campo: str) -> str:
    valores = []
    for item in registros:
        try:
            valores.append(float(item.get(campo, "")))
        except (TypeError, ValueError):
            continue

    if not valores:
        return "N/D"
    return f"{min(valores):.4f}"


if __name__ == "__main__":
    main()

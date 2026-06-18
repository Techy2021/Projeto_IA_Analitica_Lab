import math
import re
from typing import Any

from ai.rag.index import gerar_embedding, obter_collection


def buscar_trechos(pergunta: str, k: int = 4) -> list[dict[str, Any]]:
    pergunta = (pergunta or "").strip()
    if not pergunta:
        return []

    collection = obter_collection()
    total_indexado = collection.count()
    if total_indexado == 0:
        return []

    embedding = gerar_embedding(pergunta)
    resultado = collection.query(
        query_embeddings=[embedding],
        n_results=min(total_indexado, max(k * 4, k)),
        include=["documents", "metadatas", "distances"],
    )

    documentos = resultado.get("documents", [[]])[0]
    metadatas = resultado.get("metadatas", [[]])[0]
    distancias = resultado.get("distances", [[]])[0]

    trechos = []
    for documento, metadata, distancia in zip(documentos, metadatas, distancias):
        score = _distancia_para_score(distancia)
        score_semantico = _distancia_para_score(distancia)
        score_lexical = _score_lexical(pergunta, documento)
        score = (score_semantico or 0.0) + score_lexical
        trechos.append(
            {
                "texto": documento,
                "fonte": metadata.get("source_file", "fonte_desconhecida"),
                "caminho": metadata.get("source_path", ""),
                "chunk": metadata.get("chunk"),
                "distancia": distancia,
                "score": score,
                "score_semantico": score_semantico,
                "score_lexical": score_lexical,
            }
        )
    return sorted(trechos, key=lambda item: item.get("score") or 0.0, reverse=True)[:k]


def formatar_trechos_para_contexto(trechos: list[dict[str, Any]]) -> str:
    if not trechos:
        return (
            "Nenhum trecho relevante foi encontrado na base de conhecimento. "
            "Nao ha informacao suficiente para responder com base nos documentos."
        )

    blocos = []
    for indice, trecho in enumerate(trechos, start=1):
        score = trecho.get("score")
        score_texto = f"{score:.3f}" if isinstance(score, float) else "N/D"
        blocos.append(
            f"[Trecho {indice} | fonte: {trecho.get('fonte')} | "
            f"chunk: {trecho.get('chunk')} | score: {score_texto}]\n"
            f"{trecho.get('texto')}"
        )
    return "\n\n".join(blocos)


def _distancia_para_score(distancia: Any) -> float | None:
    if distancia is None:
        return None
    try:
        valor = float(distancia)
    except (TypeError, ValueError):
        return None
    if math.isnan(valor):
        return None
    return 1 / (1 + max(valor, 0.0))


def _score_lexical(pergunta: str, texto: str) -> float:
    termos_pergunta = _termos(pergunta)
    if not termos_pergunta:
        return 0.0
    termos_texto = set(_termos(texto))
    encontrados = sum(1 for termo in termos_pergunta if termo in termos_texto)
    return encontrados / len(termos_pergunta)


def _termos(texto: str) -> list[str]:
    stopwords = {
        "a",
        "as",
        "ao",
        "com",
        "da",
        "de",
        "do",
        "e",
        "em",
        "foi",
        "foram",
        "na",
        "no",
        "o",
        "os",
        "para",
        "qual",
        "quais",
        "que",
        "um",
        "uma",
    }
    termos = re.findall(r"[a-zA-ZÀ-ÿ0-9_]{3,}", texto.lower())
    return [termo for termo in termos if termo not in stopwords]

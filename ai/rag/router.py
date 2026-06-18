import json
import time
from datetime import datetime

from ai.intent_router import identificar_intencao
from ai.numeric_query_engine import (
    executar_consulta_numerica,
    formatar_resposta_consulta,
)
from app.config import REPORTS_DIR, criar_pastas


NUMERIC_LOG_PATH = REPORTS_DIR / "rag_numeric_router.jsonl"


def classificar_pergunta(pergunta: str) -> str:
    """Mantem o contrato antigo sobre o novo roteador de intencoes."""
    intencao = identificar_intencao(pergunta)["intencao"]
    if intencao == "consulta_numerica":
        return "numerica"
    if intencao in {"metricas_modelo", "colunas_treinamento"}:
        return "treinamento"
    if intencao in {"interpretacao_laboratorial", "limitacoes_modelo"}:
        return "conceitual"
    return "geral"


def responder_pergunta_numerica(
    pergunta: str,
    tolerancia: float = 0.2,
) -> dict:
    """Executa o novo motor e inclui aliases usados por consumidores antigos."""
    inicio = time.perf_counter()
    roteamento = identificar_intencao(pergunta)
    try:
        if roteamento["intencao"] != "consulta_numerica":
            return {
                "status": "nao_reconhecida",
                "tipo": classificar_pergunta(pergunta),
                "mensagem": "Pergunta numerica ainda nao suportada nesta versao.",
            }

        resultado = executar_consulta_numerica(
            roteamento,
            tolerancia=tolerancia,
        )
        resultado["tipo"] = "numerica"
        resultado["coluna_consultada"] = (
            resultado.get("variavel_alvo") or resultado.get("filtro_variavel")
        )
        resultado["total_registros"] = resultado.get("total_amostras", 0)
        resultado["media_valor"] = (
            resultado.get("resultado")
            if resultado.get("operacao") == "media"
            else None
        )
        resultado["faixa_usada"] = resultado.get("filtro_utilizado")
        return resultado
    finally:
        _registrar_log(
            {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "pergunta": pergunta,
                "tipo": classificar_pergunta(pergunta),
                "intencao": roteamento.get("intencao"),
                "tempo_consulta_ms": int((time.perf_counter() - inicio) * 1000),
            }
        )


def formatar_resposta_numerica(resultado: dict) -> str:
    """Formata tanto resultados novos quanto mensagens de compatibilidade."""
    if resultado.get("status") not in {"ok", "sem_dados"}:
        return resultado.get(
            "mensagem",
            "Nao foi possivel responder a pergunta numerica com os dados disponiveis.",
        )
    return formatar_resposta_consulta(resultado)


def _registrar_log(registro: dict) -> None:
    criar_pastas()
    NUMERIC_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with NUMERIC_LOG_PATH.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")

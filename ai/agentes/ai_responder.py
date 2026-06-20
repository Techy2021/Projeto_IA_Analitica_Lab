import json
import re
from typing import Any

import pandas as pd
from ai.intent_router import identificar_intencao
from ai.llm_provider import gerar_resposta_llm, obter_configuracao_llm
from ai.numeric_query_engine import (
    executar_consulta_numerica,
    formatar_resposta_consulta,
)
from app.config import METADATA_MODEL_PATH, REPORTS_DIR


def responder_pergunta(pergunta: str) -> dict[str, Any]:
    """Executa o fluxo completo e normaliza o retorno para consumidores e testes."""
    from ai.agentes.crewai_agents_lab import executar_crew_lab

    resultado = executar_crew_lab(pergunta)
    contexto = _extrair_contexto(resultado)
    return {
        "pergunta": pergunta,
        "resposta": resultado.get("resposta", ""),
        "contexto": contexto,
        "resultado_bruto": resultado,
    }


def _extrair_contexto(resultado: dict[str, Any]) -> str:
    """Resume fontes, ferramentas e agentes usados sem expor objetos internos."""
    partes = []

    if resultado.get("resultado_numerico"):
        partes.append(
            "Resultado numerico usado:\n"
            + json.dumps(
                resultado["resultado_numerico"],
                ensure_ascii=False,
                default=str,
                indent=2,
            )
        )

    ferramentas = resultado.get("ferramenta_utilizada") or []
    if ferramentas:
        partes.append("Ferramentas utilizadas: " + ", ".join(map(str, ferramentas)))

    agentes = resultado.get("agentes") or []
    if agentes:
        partes.append("Agentes utilizados: " + ", ".join(map(str, agentes)))

    if resultado.get("llm_model"):
        partes.append(
            "LLM: "
            f"{resultado.get('llm_provider', 'indefinido')} / "
            f"{resultado['llm_model']}"
        )
    elif resultado.get("modelo_ollama"):
        partes.append(f"Modelo LLM: {resultado['modelo_ollama']}")

    if resultado.get("status"):
        partes.append(f"Status da execucao: {resultado['status']}")

    return "\n\n".join(partes)


def responder_pergunta_teste(pergunta: str) -> dict[str, Any]:
    """Resposta rapida e determinista; usa Ollama apenas como ultimo fallback."""
    pergunta = (pergunta or "").strip()
    pergunta_normalizada = _normalizar(pergunta)

    if not pergunta:
        return _montar_resposta_teste(
            pergunta,
            "Pergunta vazia. Nao ha dado suficiente para responder.",
            "validacao_local",
        )

    roteamento = identificar_intencao(pergunta)
    intencao = roteamento["intencao"]

    if intencao == "colunas_treinamento":
        resposta, contexto = _responder_colunas_modelo()
        return _montar_resposta_teste(pergunta, resposta, contexto)

    if intencao == "metricas_modelo":
        resposta, contexto = _responder_metricas_modelo()
        return _montar_resposta_teste(pergunta, resposta, contexto)

    if intencao == "consulta_numerica":
        resultado = executar_consulta_numerica(roteamento)
        return _montar_resposta_teste(
            pergunta,
            _limitar_linhas(formatar_resposta_consulta(resultado)),
            json.dumps(resultado, ensure_ascii=False, default=str),
        )

    if intencao == "fora_escopo":
        return _montar_resposta_teste(
            pergunta,
            "Esta plataforma responde apenas perguntas sobre dados laboratoriais, "
            "modelos, qualidade de soja e farelo e documentos tecnicos relacionados.",
            "resposta_segura_fora_escopo",
        )

    if intencao == "limitacoes_modelo":
        return _montar_resposta_teste(
            pergunta,
            "O modelo e apoio a decisao. Nao deve ser usado fora da faixa de "
            "treinamento, com entradas nao validadas ou sem revisao humana.",
            "resposta_local_limitacoes",
        )

    resposta_conceitual = _responder_conceitual_curto(pergunta_normalizada)
    if resposta_conceitual:
        return _montar_resposta_teste(
            pergunta,
            resposta_conceitual,
            "resposta_conceitual_local",
        )

    resposta_llm = _responder_com_llm_curto(pergunta)
    config_llm = obter_configuracao_llm()
    return _montar_resposta_teste(
        pergunta,
        resposta_llm,
        (
            f"llm_provider={config_llm.provedor}; "
            f"llm_model={config_llm.modelo}; contexto_reduzido=true"
        ),
    )


def _montar_resposta_teste(pergunta: str, resposta: str, contexto: str) -> dict[str, Any]:
    return {
        "pergunta": pergunta,
        "resposta": _limitar_linhas(resposta),
        "contexto": contexto,
        "modelo_teste": obter_configuracao_llm().modelo,
        "modo": "fast_local",
    }


def _responder_colunas_modelo() -> tuple[str, str]:
    metadata = _carregar_metadata()
    colunas = metadata.get("colunas_usadas") or []
    alvo = metadata.get("alvo", "N/D")

    if not colunas:
        return (
            "Nao ha colunas de treinamento registradas nos metadados do modelo.",
            "metadata_modelo.json sem colunas_usadas",
        )

    primeiras = ", ".join(map(str, colunas[:12]))
    complemento = f" Outras colunas: {len(colunas) - 12}." if len(colunas) > 12 else ""
    resposta = (
        f"O alvo do modelo e {alvo}. As variaveis preditoras registradas incluem: "
        f"{primeiras}.{complemento} Nao ha inferencia de colunas fora de metadata_modelo.json."
    )
    return resposta, "metadata_modelo.json: colunas_usadas e alvo"


def _responder_metricas_modelo() -> tuple[str, str]:
    """Formata metricas persistidas, priorizando o CSV da ultima execucao."""
    metadata = _carregar_metadata()
    metricas_metadata = metadata.get("metricas") or {}
    metricas_csv = _carregar_metricas_csv()
    metricas = metricas_csv or metricas_metadata

    if not metricas:
        return (
            "Nao ha metricas registradas para avaliar o modelo atual.",
            "metricas_modelo.csv e metadata_modelo.json sem metricas",
        )

    partes = [f"{nome}={float(valor):.4f}" for nome, valor in metricas.items()]
    resposta = (
        "Metricas do modelo FLAML atual: "
        + ", ".join(partes)
        + ". Elas medem desempenho em teste e nao autorizam liberacao automatica; "
        "a decisao deve considerar validacao, faixa de aplicacao e criterio do laboratorio."
    )
    return resposta, "reports/metricas_modelo.csv; models/metadata_modelo.json"


def _responder_conceitual_curto(pergunta_normalizada: str) -> str | None:
    if "liberar automaticamente" in pergunta_normalizada or "liberar" in pergunta_normalizada:
        return (
            "Nao. O modelo nao deve liberar lote automaticamente. Ele e apoio a decisao; "
            "o analista deve verificar dados, rastreabilidade, criterios oficiais e resultados laboratoriais."
        )

    if all(termo in pergunta_normalizada for termo in ["previsao", "calculo", "interpretacao"]):
        return (
            "Previsao e a saida do modelo treinado. Calculo e uma estatistica obtida dos dados. "
            "Interpretacao e julgamento tecnico considerando metodo, limites, historico e criterio laboratorial."
        )

    if "urease" in pergunta_normalizada:
        return (
            "Urease alta pode indicar tratamento termico insuficiente no farelo de soja. "
            "A interpretacao exige limites internos, metodo analitico e revisao tecnica; nao implica liberacao automatica."
        )

    if "solubilidade" in pergunta_normalizada or "koh" in pergunta_normalizada:
        return (
            "Solubilidade KOH baixa pode sugerir dano termico e menor qualidade proteica. "
            "A conclusao depende do metodo, limites internos e outros resultados como urease, umidade e proteina."
        )

    if "confiar" in pergunta_normalizada or "cuidados" in pergunta_normalizada:
        return (
            "Antes de confiar na predicao, valide entradas, faixa de treinamento, metricas, rastreabilidade "
            "e coerencia com criterios laboratoriais. O modelo apoia, mas nao substitui revisao humana."
        )

    return None


def _responder_com_llm_curto(pergunta: str) -> str:
    prompt = (
        "Responda em portugues tecnico para laboratorio, em no maximo 8 linhas. "
        "Nao invente valores. Nao recomende liberacao automatica de lotes. "
        "Se faltarem dados, diga claramente.\n\n"
        f"Pergunta: {pergunta}"
    )
    return gerar_resposta_llm(prompt)


def _carregar_metadata() -> dict[str, Any]:
    """Le metadados para respostas rapidas sem carregar o artefato do modelo."""
    if not METADATA_MODEL_PATH.exists():
        return {}
    return json.loads(METADATA_MODEL_PATH.read_text(encoding="utf-8"))


def _carregar_metricas_csv() -> dict[str, float]:
    """Converte o CSV de metricas no formato chave/valor usado pelo chat."""
    caminho = REPORTS_DIR / "metricas_modelo.csv"
    if not caminho.exists():
        return {}
    df = pd.read_csv(caminho)
    if not {"metrica", "valor"}.issubset(df.columns):
        return {}
    return {
        str(linha["metrica"]): float(linha["valor"])
        for _, linha in df.iterrows()
        if pd.notna(linha.get("metrica")) and pd.notna(linha.get("valor"))
    }


def _pergunta_sobre_colunas(pergunta_normalizada: str) -> bool:
    return "colunas" in pergunta_normalizada and (
        "treinamento" in pergunta_normalizada or "modelo" in pergunta_normalizada
    )


def _pergunta_sobre_metricas(pergunta_normalizada: str) -> bool:
    return any(
        termo in pergunta_normalizada
        for termo in ["metrica", "metricas", "mae", "rmse", "accuracy", "precision", "recall", "r2"]
    )


def _limitar_linhas(texto: str, max_linhas: int = 8) -> str:
    linhas = [linha.rstrip() for linha in (texto or "").splitlines()]
    linhas = [linha for linha in linhas if linha]
    return "\n".join(linhas[:max_linhas])


def _normalizar(texto: str) -> str:
    substituicoes = str.maketrans(
        "áàâãäéèêëíìîïóòôõöúùûüç",
        "aaaaaeeeeiiiiooooouuuuc",
    )
    return re.sub(r"\s+", " ", (texto or "").lower().translate(substituicoes)).strip()

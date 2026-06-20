"""Execução controlada do Assistente de IA para a interface Streamlit."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any, Callable

from ai.llm_provider import obter_configuracao_llm, sanitizar_texto


LOGGER = logging.getLogger(__name__)

MENSAGEM_ERRO_ASSISTENTE = (
    "Não foi possível concluir a resposta da IA. Verifique a configuração "
    "do provedor e tente novamente."
)
MENSAGEM_TIMEOUT_ASSISTENTE = (
    "O provedor de IA demorou mais que 60 segundos para responder. "
    "Verifique o modelo, a conexão e tente novamente."
)
MENSAGEM_RESPOSTA_VAZIA_ASSISTENTE = (
    "O provedor de IA retornou uma resposta vazia. Verifique modelo, chave "
    "e provedor configurado."
)


def executar_assistente_com_timeout(
    pergunta: str,
    llm_config: dict[str, Any] | None,
    *,
    timeout_segundos: int = 60,
    executor_func: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Executa o fluxo dos agentes e devolve o controle dentro do prazo."""
    if executor_func is None:
        from ai.agentes.crewai_agents_lab import executar_crew_lab

        executor_func = executar_crew_lab

    config_execucao = dict(llm_config or {})
    config_execucao["LLM_TIMEOUT"] = str(timeout_segundos)
    config_execucao["ASSISTANT_DIRECT_MODE"] = "true"
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="assistente-ia")
    futuro = executor.submit(
        executor_func,
        pergunta,
        llm_config=config_execucao,
    )
    try:
        resultado = futuro.result(timeout=timeout_segundos)
        if resultado.get("status") != "ok":
            config = obter_configuracao_llm(config_execucao)
            LOGGER.error(
                "Assistente retornou erro. provedor=%s modelo=%s detalhe=%s",
                config.provedor,
                config.modelo,
                sanitizar_texto(
                    resultado.get("traceback") or resultado.get("resposta"),
                    config,
                ),
            )
        return resultado
    except FutureTimeout as erro:
        futuro.cancel()
        LOGGER.error(
            "Timeout no Assistente de IA após %ss. provedor=%s",
            timeout_segundos,
            config_execucao.get("LLM_PROVIDER", "ambiente"),
        )
        raise TimeoutError(MENSAGEM_TIMEOUT_ASSISTENTE) from erro
    except BaseException as erro:
        if isinstance(erro, (KeyboardInterrupt, SystemExit)):
            raise
        config = obter_configuracao_llm(config_execucao)
        LOGGER.error(
            "Falha técnica no Assistente de IA. tipo=%s detalhe=%s",
            type(erro).__name__,
            sanitizar_texto(erro, config),
        )
        raise RuntimeError(MENSAGEM_ERRO_ASSISTENTE) from erro
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def preparar_resposta_assistente(resultado: dict[str, Any]) -> tuple[str, bool]:
    """Converte o retorno interno em texto seguro e estado de sucesso."""
    resposta = str(resultado.get("resposta") or "").strip()
    if not resposta:
        return MENSAGEM_RESPOSTA_VAZIA_ASSISTENTE, False
    if resultado.get("status") != "ok":
        return MENSAGEM_ERRO_ASSISTENTE, False
    return resposta, True

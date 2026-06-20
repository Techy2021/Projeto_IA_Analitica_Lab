"""Configuracao central e chamadas aos provedores de LLM.

As credenciais podem vir do ambiente ou da memoria da sessao da interface.
Este modulo nunca persiste chaves e evita inclui-las em mensagens de erro.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from dotenv import load_dotenv


PROVEDOR_OLLAMA = "ollama"
PROVEDOR_OPENAI = "openai"
PROVEDOR_OUTRO = "custom"
PROVEDORES_SUPORTADOS = {PROVEDOR_OLLAMA, PROVEDOR_OPENAI, PROVEDOR_OUTRO}

MENSAGEM_CHAVE_OPENAI_AUSENTE = (
    "O provedor online foi selecionado, mas a chave OPENAI_API_KEY não está "
    "configurada no ambiente."
)
MENSAGEM_CHAVE_CUSTOM_AUSENTE = (
    "O provedor online foi selecionado, mas a chave de API não está configurada. "
    "Informe a chave na sessão ou configure OTHER_LLM_API_KEY no ambiente de execução."
)
MENSAGEM_FALHA_SEGURA = (
    "O provedor de IA não respondeu. Verifique a configuração e tente novamente."
)
MENSAGEM_OLLAMA_INDISPONIVEL = (
    "O provedor Ollama foi selecionado, mas não foi possível conectar ao serviço local."
)


@dataclass(frozen=True)
class ConfiguracaoLLM:
    provedor: str
    modelo: str
    base_url: str | None = None
    api_key: str | None = None
    timeout: int = 60
    origem: str = "ambiente"

    def publico(self) -> dict[str, Any]:
        """Retorna apenas dados seguros para interface, logs e traces."""
        dados = asdict(self)
        dados.pop("api_key", None)
        dados["chave_configurada"] = bool(self.api_key)
        return dados


def _valor(overrides: Mapping[str, Any], nome: str, padrao: Any = None) -> Any:
    valor = overrides.get(nome)
    if valor is not None and str(valor).strip() != "":
        return valor
    return os.getenv(nome, padrao)


def obter_configuracao_llm(
    overrides: Mapping[str, Any] | None = None,
) -> ConfiguracaoLLM:
    """Resolve configuracao da sessao, ambiente e defaults, nessa ordem."""
    load_dotenv()
    overrides = overrides or {}
    provedor = str(_valor(overrides, "LLM_PROVIDER", PROVEDOR_OLLAMA)).lower().strip()
    aliases = {
        "ollama local": PROVEDOR_OLLAMA,
        "openai api": PROVEDOR_OPENAI,
        "outro": PROVEDOR_OUTRO,
        "outro provedor via api": PROVEDOR_OUTRO,
    }
    provedor = aliases.get(provedor, provedor)
    origem = "sessao" if overrides else "ambiente"
    timeout = int(_valor(overrides, "LLM_TIMEOUT", "60"))

    if provedor == PROVEDOR_OPENAI:
        return ConfiguracaoLLM(
            provedor=provedor,
            modelo=str(_valor(overrides, "OPENAI_MODEL", "gpt-4o-mini")).strip(),
            base_url=_normalizar_url(_valor(overrides, "OPENAI_BASE_URL")),
            api_key=_segredo(_valor(overrides, "OPENAI_API_KEY")),
            timeout=timeout,
            origem=origem,
        )

    if provedor == PROVEDOR_OUTRO:
        return ConfiguracaoLLM(
            provedor=provedor,
            modelo=str(_valor(overrides, "OTHER_LLM_MODEL", "")).strip(),
            base_url=_normalizar_url(_valor(overrides, "OTHER_LLM_BASE_URL")),
            api_key=_segredo(_valor(overrides, "OTHER_LLM_API_KEY")),
            timeout=timeout,
            origem=origem,
        )

    if provedor == PROVEDOR_OLLAMA:
        return ConfiguracaoLLM(
            provedor=PROVEDOR_OLLAMA,
            modelo=str(_valor(overrides, "OLLAMA_MODEL", "gemma3:1b")).strip(),
            base_url=_normalizar_url(
                _valor(overrides, "OLLAMA_BASE_URL", "http://localhost:11434")
            ),
            timeout=timeout,
            origem=origem,
        )

    return ConfiguracaoLLM(
        provedor=provedor,
        modelo="",
        timeout=timeout,
        origem=origem,
    )


def validar_configuracao(config: ConfiguracaoLLM) -> str | None:
    if config.provedor not in PROVEDORES_SUPORTADOS:
        return (
            f"Provedor de IA inválido: {config.provedor}. "
            "Use ollama, openai ou custom."
        )
    if not config.modelo:
        return "O modelo do provedor de IA não foi configurado."
    if config.provedor == PROVEDOR_OPENAI and not config.api_key:
        return MENSAGEM_CHAVE_OPENAI_AUSENTE
    if config.provedor == PROVEDOR_OUTRO:
        if not config.base_url:
            return "Configure OTHER_LLM_BASE_URL para o provedor via API."
        if not config.api_key:
            return MENSAGEM_CHAVE_CUSTOM_AUSENTE
    return None
def criar_llm_crewai(
    overrides: Mapping[str, Any] | None = None,
    *,
    max_tokens: int = 500,
):
    """Cria o adaptador CrewAI/LiteLLM para o provedor ativo."""
    from crewai import LLM

    config = obter_configuracao_llm(overrides)
    erro = validar_configuracao(config)
    if erro:
        raise RuntimeError(erro)

    parametros: dict[str, Any] = {
        "model": _modelo_litellm(config),
        "timeout": config.timeout,
        "max_tokens": max_tokens,
    }
    if config.base_url:
        parametros["base_url"] = config.base_url
    if config.api_key:
        parametros["api_key"] = config.api_key

    try:
        return LLM(**parametros)
    except TypeError:
        if "base_url" in parametros:
            parametros["api_base"] = parametros.pop("base_url")
        return LLM(**parametros)


def gerar_resposta_llm(
    prompt: str,
    contexto: str | None = None,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> str:
    """Gera texto via LiteLLM e retorna fallback seguro em caso de falha."""
    config = obter_configuracao_llm(overrides)
    erro = validar_configuracao(config)
    if erro:
        return erro

    mensagens = [
        {
            "role": "system",
            "content": (
                "Responda em português técnico e claro. Não invente dados e deixe "
                "explícito quando o contexto for insuficiente."
            ),
        }
    ]
    if contexto:
        mensagens.append({"role": "system", "content": f"Contexto:\n{contexto}"})
    mensagens.append({"role": "user", "content": prompt})

    try:
        from litellm import completion

        parametros: dict[str, Any] = {
            "model": _modelo_litellm(config),
            "messages": mensagens,
            "timeout": config.timeout,
            "temperature": 0.1,
            "max_tokens": 500,
        }
        if config.base_url:
            parametros["api_base"] = config.base_url
        if config.api_key:
            parametros["api_key"] = config.api_key
        resposta = completion(**parametros)
        texto = _extrair_texto(resposta)
        return texto or "O provedor de IA retornou uma resposta vazia."
    except Exception:
        if config.provedor == PROVEDOR_OLLAMA:
            return MENSAGEM_OLLAMA_INDISPONIVEL
        return MENSAGEM_FALHA_SEGURA


def testar_conexao_llm(
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Executa uma chamada curta e retorna somente metadados sem segredos."""
    config = obter_configuracao_llm(overrides)
    erro = validar_configuracao(config)
    if erro:
        return {"status": "erro", "mensagem": erro, **config.publico()}

    resposta = gerar_resposta_llm(
        "Responda somente com OK.",
        overrides=overrides,
    )
    if resposta in {
        MENSAGEM_FALHA_SEGURA,
        MENSAGEM_OLLAMA_INDISPONIVEL,
        "O provedor de IA retornou uma resposta vazia.",
    }:
        return {
            "status": "erro",
            "mensagem": resposta,
            **config.publico(),
        }
    return {
        "status": "ok",
        "mensagem": "Conexão com o provedor de IA realizada com sucesso.",
        "resposta_teste": resposta[:120],
        **config.publico(),
    }


def sanitizar_texto(texto: Any, config: ConfiguracaoLLM | None = None) -> str:
    """Remove credenciais conhecidas e formatos comuns de token."""
    resultado = str(texto or "")
    segredos = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("OTHER_LLM_API_KEY"),
        config.api_key if config else None,
    ]
    for segredo in segredos:
        if segredo:
            resultado = resultado.replace(segredo, "[REDACTED]")
    resultado = re.sub(r"\bsk-[A-Za-z0-9_-]{8,}\b", "[REDACTED]", resultado)
    resultado = re.sub(
        r"(?i)(api[_ -]?key\s*[=:]\s*)\S+",
        r"\1[REDACTED]",
        resultado,
    )
    return resultado


def _modelo_litellm(config: ConfiguracaoLLM) -> str:
    if "/" in config.modelo:
        return config.modelo
    if config.provedor == PROVEDOR_OLLAMA:
        return f"ollama/{config.modelo}"
    if config.provedor in {PROVEDOR_OPENAI, PROVEDOR_OUTRO}:
        return f"openai/{config.modelo}"
    return config.modelo


def _extrair_texto(resposta: Any) -> str:
    try:
        return str(resposta.choices[0].message.content).strip()
    except (AttributeError, IndexError, TypeError):
        pass
    if isinstance(resposta, dict):
        escolhas = resposta.get("choices") or []
        if escolhas:
            mensagem = escolhas[0].get("message") or {}
            return str(mensagem.get("content") or "").strip()
    return ""


def _normalizar_url(valor: Any) -> str | None:
    texto = str(valor or "").strip()
    return texto.rstrip("/") if texto else None


def _segredo(valor: Any) -> str | None:
    texto = str(valor or "").strip()
    return texto or None

"""Configuracao central e chamadas aos provedores de LLM.

As credenciais podem vir do ambiente ou da memoria da sessao da interface.
Este modulo nunca persiste chaves e evita inclui-las em mensagens de erro.
"""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import requests
from dotenv import load_dotenv


PROVEDOR_OLLAMA = "ollama"
PROVEDOR_OPENAI = "openai"
PROVEDOR_GEMINI = "gemini"
PROVEDOR_OUTRO = "custom"
PROVEDORES_SUPORTADOS = {
    PROVEDOR_OLLAMA,
    PROVEDOR_OPENAI,
    PROVEDOR_GEMINI,
    PROVEDOR_OUTRO,
}

MENSAGEM_CHAVE_OPENAI_AUSENTE = (
    "O provedor online foi selecionado, mas a chave OPENAI_API_KEY não está "
    "configurada no ambiente."
)
MENSAGEM_CHAVE_GEMINI_AUSENTE = "Chave GEMINI_API_KEY não configurada."
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
        "google gemini": PROVEDOR_GEMINI,
        "google ai studio": PROVEDOR_GEMINI,
        "gemini api": PROVEDOR_GEMINI,
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

    if provedor == PROVEDOR_GEMINI:
        return ConfiguracaoLLM(
            provedor=provedor,
            modelo=str(
                _valor(overrides, "GEMINI_MODEL", "gemini-2.5-flash")
            ).strip(),
            api_key=_segredo(_valor(overrides, "GEMINI_API_KEY")),
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
        modelo_ollama = str(
            _valor(overrides, "OLLAMA_MODEL", "gemma3:1b")
        ).strip()
        return ConfiguracaoLLM(
            provedor=PROVEDOR_OLLAMA,
            modelo=_modelo_ollama_para_chat(modelo_ollama),
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
            "Use ollama, openai, gemini ou custom."
        )
    if not config.modelo:
        return "O modelo do provedor de IA não foi configurado."
    if config.provedor == PROVEDOR_OPENAI and not config.api_key:
        return MENSAGEM_CHAVE_OPENAI_AUSENTE
    if config.provedor == PROVEDOR_GEMINI and not config.api_key:
        return MENSAGEM_CHAVE_GEMINI_AUSENTE
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
    if config.provedor == PROVEDOR_OLLAMA:
        parametros["extra_body"] = {"think": False}
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

        modelo_thinking = (
            config.provedor == PROVEDOR_OLLAMA
            and _modelo_ollama_thinking(config.modelo)
        )
        parametros: dict[str, Any] = {
            "model": _modelo_litellm(config),
            "messages": mensagens,
            "timeout": max(config.timeout, 300) if modelo_thinking else config.timeout,
            "temperature": 0.1,
            "max_tokens": 1200 if modelo_thinking else 500,
        }
        if config.provedor == PROVEDOR_OLLAMA:
            # Modelos Qwen 3 podem consumir todo o limite em raciocínio interno
            # e devolver conteúdo vazio. O chat da aplicação precisa da resposta.
            parametros["think"] = False
            parametros["num_retries"] = 0
        if config.base_url:
            parametros["api_base"] = config.base_url
        if config.api_key:
            parametros["api_key"] = config.api_key
        resposta = completion(**parametros)
        texto = _extrair_texto(resposta)
        if texto:
            return texto
        if modelo_thinking:
            return (
                f"O modelo {config.modelo} consumiu o limite em raciocínio interno "
                "sem produzir a resposta final. Para chat, use "
                f"{_modelo_ollama_instruct(config.modelo)}."
            )
        return "O provedor de IA retornou uma resposta vazia."
    except Exception as erro:
        if config.provedor == PROVEDOR_OLLAMA:
            if "timeout" in str(erro).lower() or "timed out" in str(erro).lower():
                return (
                    "O Ollama está em execução, mas o modelo demorou mais que "
                    f"{config.timeout} segundos para responder. Tente novamente; "
                    "a primeira resposta pode levar mais tempo enquanto o modelo carrega."
                )
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

    if config.provedor == PROVEDOR_OLLAMA:
        return _testar_conexao_ollama(config)

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


def _testar_conexao_ollama(config: ConfiguracaoLLM) -> dict[str, Any]:
    """Valida serviço e modelo sem forçar o carregamento demorado do LLM."""
    try:
        resposta = requests.get(f"{config.base_url}/api/tags", timeout=5)
        resposta.raise_for_status()
        payload = resposta.json()
    except requests.exceptions.Timeout:
        return {
            "status": "erro",
            "mensagem": (
                "O Ollama não respondeu dentro de 5 segundos. Verifique a URL base "
                "e se o serviço está em execução."
            ),
            **config.publico(),
        }
    except (requests.exceptions.RequestException, ValueError):
        return {
            "status": "erro",
            "mensagem": MENSAGEM_OLLAMA_INDISPONIVEL,
            **config.publico(),
        }

    modelos = {
        str(item.get("name") or item.get("model") or "")
        for item in payload.get("models", [])
    }
    if config.modelo not in modelos:
        return {
            "status": "erro",
            "mensagem": (
                f"O Ollama está acessível, mas o modelo {config.modelo} não foi "
                f"encontrado. Execute: ollama pull {config.modelo}"
            ),
            **config.publico(),
        }

    return {
        "status": "ok",
        "mensagem": "Conexão com o Ollama e modelo verificadas com sucesso.",
        **config.publico(),
    }


def sanitizar_texto(texto: Any, config: ConfiguracaoLLM | None = None) -> str:
    """Remove credenciais conhecidas e formatos comuns de token."""
    resultado = str(texto or "")
    segredos = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("GEMINI_API_KEY"),
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
    if config.provedor == PROVEDOR_GEMINI:
        return (
            config.modelo
            if config.modelo.startswith("gemini/")
            else f"gemini/{config.modelo}"
        )
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


def _modelo_ollama_thinking(modelo: str) -> bool:
    nome = modelo.lower()
    return (
        "-thinking" in nome
        or nome in {"qwen3-vl:2b", "qwen3-vl:4b", "qwen3-vl:8b"}
    )


def _modelo_ollama_para_chat(modelo: str) -> str:
    """Troca aliases Qwen3-VL thinking pela variante adequada para chat."""
    if modelo.lower() in {"qwen3-vl:2b", "qwen3-vl:4b", "qwen3-vl:8b"}:
        return f"{modelo}-instruct"
    return modelo


def _modelo_ollama_instruct(modelo: str) -> str:
    nome = modelo.lower()
    if "-thinking" in nome:
        return re.sub(r"-thinking(?=-|$)", "-instruct", modelo, count=1)
    if nome in {"qwen3-vl:2b", "qwen3-vl:4b", "qwen3-vl:8b"}:
        return f"{modelo}-instruct"
    return modelo


def _segredo(valor: Any) -> str | None:
    texto = str(valor or "").strip()
    return texto or None

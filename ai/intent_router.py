import json
import re
import time
import unicodedata
from datetime import datetime
from typing import Any

from app.config import REPORTS_DIR, criar_pastas


INTENT_LOG_PATH = REPORTS_DIR / "intent_router.jsonl"

VARIABLES = {
    "materia mineral": ("farelo_materia_mineral_pct", "materia mineral do farelo"),
    "insoluveis": ("farelo_insoluveis_hcl_pct", "insoluveis HCL do farelo"),
    "solubilidade": ("farelo_solubilidade_koh_pct", "solubilidade KOH do farelo"),
    "proteina": ("farelo_proteina_pct", "proteina do farelo"),
    "umidade": ("farelo_umidade_pct", "umidade do farelo"),
    "fibras": ("farelo_fibras_pct", "fibras do farelo"),
    "fibra": ("farelo_fibras_pct", "fibras do farelo"),
    "urease": ("farelo_urease_delta_ph", "urease do farelo"),
    "oleo": ("farelo_oleo_pct", "oleo residual do farelo"),
    "koh": ("farelo_solubilidade_koh_pct", "solubilidade KOH do farelo"),
}

OPERATION_PATTERNS = (
    ("contagem", ("quantas amostras", "quantos registros", "quantidade", "contagem")),
    ("media", ("media", "valor medio")),
    ("mediana", ("mediana",)),
    ("maximo", ("maior valor", "valor maximo", "maximo")),
    ("minimo", ("menor valor", "valor minimo", "minimo")),
)

OUT_OF_SCOPE_TERMS = (
    "previsao do tempo",
    "clima",
    "quem ganhou",
    "jogo",
    "futebol",
    "cotacao do dolar",
    "cotacao",
    "dolar",
    "bolsa de valores",
)


def identificar_intencao(pergunta: str) -> dict[str, Any]:
    """Extrai intencao e entidades sem chamar LLM ou acessar o banco."""
    inicio = time.perf_counter()
    texto = normalizar_texto(pergunta)
    entidades: dict[str, Any] = {}

    if any(termo in texto for termo in OUT_OF_SCOPE_TERMS):
        return _resultado("fora_escopo", 0.99, entidades, inicio)

    if _contem_metricas(texto):
        return _resultado("metricas_modelo", 0.98, entidades, inicio)

    if _contem_colunas_treinamento(texto):
        return _resultado("colunas_treinamento", 0.97, entidades, inicio)

    if _contem_limitacoes(texto):
        return _resultado("limitacoes_modelo", 0.94, entidades, inicio)

    if _contem_predicao(texto):
        entidades["valores_informados"] = _extrair_todos_valores(texto)
        return _resultado("predicao_amostra", 0.93, entidades, inicio)

    operacao = _identificar_operacao(texto)
    variaveis = _encontrar_variaveis(texto)
    if operacao and variaveis:
        entidades = _extrair_entidades_numericas(texto, operacao, variaveis)
        confianca = 0.96 if entidades.get("variavel_alvo") else 0.88
        return _resultado("consulta_numerica", confianca, entidades, inicio)

    if _contem_interpretacao_laboratorial(texto):
        return _resultado("interpretacao_laboratorial", 0.91, entidades, inicio)

    return _resultado("geral_laboratorial", 0.55, entidades, inicio)


def registrar_intencao(
    pergunta: str,
    roteamento: dict[str, Any],
    ferramenta: str,
    tempo_execucao_ms: int,
    status: str,
) -> None:
    """Registra a decisao de roteamento em JSONL para auditoria."""
    criar_pastas()
    registro = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "pergunta": pergunta,
        "intencao": roteamento.get("intencao"),
        "confianca": roteamento.get("confianca"),
        "entidades": roteamento.get("entidades", {}),
        "ferramenta": ferramenta,
        "tempo_execucao_ms": tempo_execucao_ms,
        "status": status,
    }
    INTENT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with INTENT_LOG_PATH.open("a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False, default=str) + "\n")


def normalizar_texto(texto: str) -> str:
    texto = (texto or "").strip().lower()
    substituicoes_encoding = {
        "m?dia": "media",
        "m?nimo": "minimo",
        "m?ximo": "maximo",
        "prote?na": "proteina",
        "previs?o": "previsao",
        "solubilidade": "solubilidade",
        "mat?ria": "materia",
        "?leo": "oleo",
        "insol?veis": "insoluveis",
        "r?": "r2",
    }
    for origem, destino in substituicoes_encoding.items():
        texto = texto.replace(origem, destino)
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(char for char in texto if not unicodedata.combining(char))
    texto = texto.replace("?", " ")
    return re.sub(r"\s+", " ", texto)


def _resultado(
    intencao: str,
    confianca: float,
    entidades: dict[str, Any],
    inicio: float,
) -> dict[str, Any]:
    return {
        "intencao": intencao,
        "confianca": confianca,
        "entidades": entidades,
        "tempo_classificacao_ms": int((time.perf_counter() - inicio) * 1000),
    }


def _contem_metricas(texto: str) -> bool:
    return any(
        termo in texto
        for termo in ("metricas", "metrica", "r2", "rmse", "mae", "accuracy", "precision", "recall", "f1")
    )


def _contem_colunas_treinamento(texto: str) -> bool:
    return (
        any(termo in texto for termo in ("colunas", "variaveis", "features"))
        and any(termo in texto for termo in ("treinamento", "treino", "alimentam o modelo", "modelo"))
    )


def _contem_limitacoes(texto: str) -> bool:
    return any(
        termo in texto
        for termo in (
            "posso confiar",
            "confiavel",
            "confiabilidade",
            "limitacoes",
            "limitacao",
            "quando o modelo nao",
            "quando nao deve",
            "cuidados com o modelo",
        )
    )


def _contem_predicao(texto: str) -> bool:
    return any(
        termo in texto
        for termo in ("preveja", "prever amostra", "classifique", "classificar esta amostra", "predicao da amostra")
    )


def _contem_interpretacao_laboratorial(texto: str) -> bool:
    variavel_laboratorial = any(alias in texto for alias in VARIABLES)
    termo_interpretacao = any(
        termo in texto
        for termo in ("o que significa", "o que indica", "como interpretar", "alta", "baixa", "baixo")
    )
    return variavel_laboratorial and termo_interpretacao


def _identificar_operacao(texto: str) -> str | None:
    for operacao, padroes in OPERATION_PATTERNS:
        if any(padrao in texto for padrao in padroes):
            return operacao
    return None


def _encontrar_variaveis(texto: str) -> list[dict[str, Any]]:
    encontradas = []
    colunas_vistas = set()
    for alias, (coluna, rotulo) in VARIABLES.items():
        for match in re.finditer(rf"\b{re.escape(alias)}\b", texto):
            chave = (match.start(), coluna)
            if chave not in colunas_vistas:
                encontradas.append(
                    {
                        "alias": alias,
                        "coluna": coluna,
                        "rotulo": rotulo,
                        "posicao": match.start(),
                    }
                )
                colunas_vistas.add(chave)
    return sorted(encontradas, key=lambda item: item["posicao"])


def _extrair_entidades_numericas(
    texto: str,
    operacao: str,
    variaveis: list[dict[str, Any]],
) -> dict[str, Any]:
    entidades: dict[str, Any] = {"operacao": operacao}

    if operacao == "contagem":
        filtro = variaveis[0]
        entidades["variavel_alvo"] = None
    else:
        posicao_operacao = _posicao_operacao(texto, operacao)
        posicao_filtro = _posicao_inicio_filtro(texto, inicio=posicao_operacao)
        candidatos_alvo = [
            variavel
            for variavel in variaveis
            if variavel["posicao"] >= posicao_operacao
            and (posicao_filtro is None or variavel["posicao"] < posicao_filtro)
        ]
        alvo = candidatos_alvo[0] if candidatos_alvo else None
        if alvo:
            entidades["variavel_alvo"] = alvo["coluna"]
            entidades["variavel_alvo_rotulo"] = alvo["rotulo"]
        else:
            entidades["variavel_alvo"] = None
            entidades["variavel_nao_mapeada"] = True

        filtro = None
        if posicao_filtro is not None:
            filtro = next(
                (
                    variavel
                    for variavel in variaveis
                    if variavel["posicao"] >= posicao_filtro
                ),
                None,
            )
        if filtro is None:
            filtro = next(
                (
                    variavel
                    for variavel in variaveis
                    if not alvo or variavel["coluna"] != alvo["coluna"]
                ),
                None,
            )

    if filtro:
        entidades["filtro_variavel"] = filtro["coluna"]
        entidades["filtro_variavel_rotulo"] = filtro["rotulo"]
        entidades.update(_extrair_condicao_filtro(texto, filtro["posicao"]))

    return entidades


def _posicao_operacao(texto: str, operacao: str) -> int:
    padroes = dict(OPERATION_PATTERNS)[operacao]
    posicoes = [texto.find(padrao) for padrao in padroes if padrao in texto]
    return min(posicoes) if posicoes else 0


def _posicao_inicio_filtro(texto: str, inicio: int = 0) -> int | None:
    posicoes = [
        texto.find(marcador)
        for marcador in (" quando ", " para amostras", " com ")
        if texto.find(marcador) >= inicio
    ]
    return min(posicoes) if posicoes else None


def _extrair_condicao_filtro(texto: str, posicao_variavel: int) -> dict[str, Any]:
    trecho = texto[posicao_variavel:]
    match_valor = re.search(r"-?\d+(?:[,.]\d+)?", trecho)
    if not match_valor:
        return {}

    valor = float(match_valor.group(0).replace(",", "."))
    prefixo = trecho[: match_valor.start()]
    if any(termo in prefixo for termo in ("acima de", "maior que", "superior a")):
        comparador = "maior_que"
    elif any(termo in prefixo for termo in ("abaixo de", "menor que", "inferior a")):
        comparador = "menor_que"
    else:
        comparador = "aproximado"
    return {"filtro_operador": comparador, "filtro_valor": valor}


def _extrair_todos_valores(texto: str) -> dict[str, float]:
    valores: dict[str, float] = {}
    for variavel in _encontrar_variaveis(texto):
        trecho = texto[variavel["posicao"] :]
        match = re.search(r"-?\d+(?:[,.]\d+)?", trecho)
        if match:
            valores[variavel["coluna"]] = float(match.group(0).replace(",", "."))
    return valores

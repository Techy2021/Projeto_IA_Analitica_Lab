import time
from typing import Any

from database.consultas import consultar_sql


ALLOWED_COLUMNS = {
    "farelo_umidade_pct",
    "farelo_proteina_pct",
    "farelo_oleo_pct",
    "farelo_fibras_pct",
    "farelo_urease_delta_ph",
    "farelo_materia_mineral_pct",
    "farelo_solubilidade_koh_pct",
    "farelo_insoluveis_hcl_pct",
}
ALLOWED_OPERATIONS = {
    "media": "AVG",
    "mediana": "MEDIAN",
    "maximo": "MAX",
    "minimo": "MIN",
}


def executar_consulta_numerica(
    roteamento: dict[str, Any],
    tolerancia: float = 0.2,
    minimo_amostras: int = 10,
) -> dict[str, Any]:
    """Executa uma consulta estatistica usando apenas colunas permitidas."""
    inicio = time.perf_counter()
    entidades = roteamento.get("entidades", {})
    operacao = entidades.get("operacao")
    alvo = entidades.get("variavel_alvo")
    filtro_coluna = entidades.get("filtro_variavel")
    filtro_operador = entidades.get("filtro_operador")
    filtro_valor = entidades.get("filtro_valor")

    try:
        _validar_entidades(operacao, alvo, filtro_coluna)
    except ValueError as erro:
        return {
            "status": "nao_suportada",
            "fonte": "duckdb",
            "intencao": "consulta_numerica",
            "operacao": operacao,
            "variavel_alvo": alvo,
            "filtro_variavel": filtro_coluna,
            "mensagem": str(erro),
            "tempo_consulta_ms": int((time.perf_counter() - inicio) * 1000),
        }
    where_sql, parametros, filtro_descricao = _montar_filtro(
        filtro_coluna,
        filtro_operador,
        filtro_valor,
        tolerancia,
    )
    tabela = _obter_tabela_dataset()

    if operacao == "contagem":
        expressao_resultado = "COUNT(*)"
        calculo_descricao = "contagem de amostras"
    else:
        expressao_resultado = f"{ALLOWED_OPERATIONS[operacao]}({alvo})"
        calculo_descricao = f"{operacao} de {alvo}"

    query = (
        "SELECT COUNT(*) AS total_amostras, "
        f"{expressao_resultado} AS resultado "
        f"FROM {tabela}{where_sql}"
    )
    df = consultar_sql(_interpolar_parametros(query, parametros))
    registro = df.iloc[0].to_dict() if not df.empty else {}
    total = int(registro.get("total_amostras") or 0)
    resultado_valor = registro.get("resultado")
    if resultado_valor is not None:
        resultado_valor = float(resultado_valor)

    status = "ok" if total > 0 else "sem_dados"
    return {
        "status": status,
        "fonte": "duckdb",
        "tabela": tabela,
        "intencao": "consulta_numerica",
        "operacao": operacao,
        "variavel_alvo": alvo,
        "variavel_alvo_rotulo": entidades.get("variavel_alvo_rotulo"),
        "filtro_variavel": filtro_coluna,
        "filtro_variavel_rotulo": entidades.get("filtro_variavel_rotulo"),
        "filtro_operador": filtro_operador,
        "filtro_valor": filtro_valor,
        "tolerancia": tolerancia if filtro_operador == "aproximado" else None,
        "filtro_utilizado": filtro_descricao,
        "calculo_realizado": calculo_descricao,
        "total_amostras": total,
        "resultado": resultado_valor,
        "unidade": _unidade_coluna(alvo or filtro_coluna),
        "poucas_amostras": 0 < total < minimo_amostras,
        "minimo_amostras_recomendado": minimo_amostras,
        "tempo_consulta_ms": int((time.perf_counter() - inicio) * 1000),
    }


def formatar_resposta_consulta(resultado: dict[str, Any]) -> str:
    """Gera uma resposta tecnica a partir do resultado calculado no DuckDB."""
    if resultado.get("status") == "nao_suportada":
        return (
            resultado.get("mensagem", "Consulta numerica nao suportada.")
            + " Informe uma variavel laboratorial mapeada."
        )
    if resultado.get("status") != "ok":
        return (
            "Nao foram encontradas amostras para o filtro solicitado. "
            "Revise a variavel, o valor ou a faixa utilizada."
        )

    total = resultado["total_amostras"]
    filtro = resultado["filtro_utilizado"]
    operacao = resultado["operacao"]
    valor = resultado["resultado"]
    alvo = resultado.get("variavel_alvo") or resultado.get("filtro_variavel")
    unidade = resultado.get("unidade") or ""
    sufixo = unidade if unidade == "%" else (f" {unidade}" if unidade else "")

    if operacao == "contagem":
        texto_resultado = f"Foram encontradas {total} amostras."
    else:
        texto_resultado = (
            f"O resultado para {operacao} de {alvo} foi {valor:.2f}{sufixo}. "
            f"Foram consideradas {total} amostras."
        )

    observacao = (
        "\n\nAtencao: ha poucas amostras para esse recorte; interprete o resultado "
        "com cautela."
        if resultado.get("poucas_amostras")
        else ""
    )
    filtro_texto = f"\nFiltro utilizado: {filtro}." if filtro else "\nFiltro utilizado: nenhum."
    return (
        f"{texto_resultado}{filtro_texto}\n"
        f"Calculo realizado no DuckDB: {resultado['calculo_realizado']}.\n\n"
        "Esse resultado representa uma consulta estatistica ao conjunto de dados "
        "disponivel e nao uma previsao do modelo."
        f"{observacao}"
    )


def _validar_entidades(
    operacao: str | None,
    alvo: str | None,
    filtro_coluna: str | None,
) -> None:
    if operacao not in {*ALLOWED_OPERATIONS, "contagem"}:
        raise ValueError("Operacao numerica nao suportada.")
    if operacao != "contagem" and alvo not in ALLOWED_COLUMNS:
        raise ValueError("Variavel alvo nao suportada para consulta numerica.")
    if filtro_coluna and filtro_coluna not in ALLOWED_COLUMNS:
        raise ValueError("Variavel de filtro nao suportada.")


def _montar_filtro(
    coluna: str | None,
    operador: str | None,
    valor: float | None,
    tolerancia: float,
) -> tuple[str, list[float], str | None]:
    if not coluna or valor is None:
        return "", [], None
    unidade = _unidade_coluna(coluna)
    sufixo = unidade if unidade == "%" else (f" {unidade}" if unidade else "")
    if operador == "maior_que":
        return (
            f" WHERE {coluna} > ?",
            [valor],
            f"{coluna} acima de {valor:g}{sufixo}",
        )
    if operador == "menor_que":
        return (
            f" WHERE {coluna} < ?",
            [valor],
            f"{coluna} abaixo de {valor:g}{sufixo}",
        )

    minimo = valor - tolerancia
    maximo = valor + tolerancia
    return (
        f" WHERE {coluna} BETWEEN ? AND ?",
        [minimo, maximo],
        f"{coluna} entre {minimo:g}{sufixo} e {maximo:g}{sufixo}",
    )


def _interpolar_parametros(query: str, parametros: list[float]) -> str:
    """Substitui apenas valores numericos gerados internamente pelo roteador."""
    for valor in parametros:
        query = query.replace("?", str(float(valor)), 1)
    return query


def _obter_tabela_dataset() -> str:
    try:
        consultar_sql("SELECT 1 FROM dataset LIMIT 1")
        return "dataset"
    except Exception:
        return "dataset_lab"


def _unidade_coluna(coluna: str | None) -> str:
    if not coluna:
        return ""
    if coluna.endswith("_pct"):
        return "%"
    if coluna == "farelo_urease_delta_ph":
        return "delta pH"
    return ""

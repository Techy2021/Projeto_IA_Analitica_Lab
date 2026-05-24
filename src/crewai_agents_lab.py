import json
import os
import re
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv

from src.config import REPORTS_DIR, criar_pastas
from src.crewai_tools_lab import (
    CREWAI_TOOLS_DISPONIVEL,
    CREWAI_TOOLS_IMPORT_ERROR,
    CREWAI_TOOLS_IMPORT_TRACEBACK,
    consultar_dados_laboratorio,
    consultar_dados_laboratorio_func,
    obter_amostra_media,
    obter_amostra_media_func,
    obter_metadata_modelo,
    obter_metadata_modelo_func,
    prever_farelo_soja,
    prever_farelo_soja_func,
)
from src.ollama_check import (
    MENSAGEM_MODELO_INDISPONIVEL,
    OLLAMA_MODEL_PADRAO,
    modelo_ollama_disponivel,
    obter_config_ollama,
    verificar_ollama,
)


try:
    from crewai import Agent, Crew, LLM, Process, Task

    CREWAI_DISPONIVEL = True
    CREWAI_IMPORT_ERROR = None
    CREWAI_IMPORT_TRACEBACK = None
except ModuleNotFoundError as erro:
    Agent = Crew = LLM = Process = Task = None
    CREWAI_DISPONIVEL = False
    CREWAI_IMPORT_ERROR = (
        f"Pacote não encontrado: {erro.name}. Execute: "
        "python -m pip install -r requirements.txt"
    )
    CREWAI_IMPORT_TRACEBACK = traceback.format_exc()
except Exception as erro:
    Agent = Crew = LLM = Process = Task = None
    CREWAI_DISPONIVEL = False
    CREWAI_IMPORT_ERROR = (
        f"Erro ao importar CrewAI: {type(erro).__name__}: {erro}"
    )
    CREWAI_IMPORT_TRACEBACK = traceback.format_exc()


AGENTE_ANALISTA = "Agente Analista de Qualidade Laboratorial"
AGENTE_ESPECIALISTA = "Agente Especialista em Modelo Preditivo"
TRACE_PATH = REPORTS_DIR / "traces_crewai_ollama.jsonl"


def _registrar_trace(registro: dict) -> None:
    criar_pastas()
    TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(TRACE_PATH, "a", encoding="utf-8") as arquivo:
        arquivo.write(json.dumps(registro, ensure_ascii=False) + "\n")


def _criar_llm_ollama():
    base_url, modelo = obter_config_ollama()
    try:
        return LLM(model=f"ollama/{modelo}", base_url=base_url)
    except TypeError:
        return LLM(model=f"ollama/{modelo}", api_base=base_url)


def rotear_tarefa_crewai(pergunta: str) -> dict:
    texto = (pergunta or "").lower()
    palavras_analista = [
        "previsão",
        "previsao",
        "prever",
        "classificar",
        "classificação",
        "classificacao",
        "amostra",
        "resultado",
        "média",
        "media",
        "distribuição",
        "distribuicao",
        "quantidade",
        "classe",
    ]
    palavras_especialista = [
        "modelo",
        "métrica",
        "metrica",
        "métricas",
        "metricas",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "r2",
        "r²",
        "mae",
        "rmse",
        "colunas",
        "treinamento",
    ]

    usar_analista = any(palavra in texto for palavra in palavras_analista)
    usar_especialista = any(palavra in texto for palavra in palavras_especialista)

    if not usar_analista and not usar_especialista:
        usar_analista = True
        usar_especialista = True

    agentes = []
    if usar_analista:
        agentes.append(AGENTE_ANALISTA)
    if usar_especialista:
        agentes.append(AGENTE_ESPECIALISTA)

    return {
        "pergunta": pergunta,
        "usar_analista": usar_analista,
        "usar_especialista": usar_especialista,
        "agentes": agentes,
        "instrucao": (
            "Use ferramentas quando necessário, responda em português técnico e claro "
            "e nunca invente resultados não retornados pelas ferramentas."
        ),
    }


def rotear_pergunta_crewai(pergunta: str) -> dict:
    return rotear_tarefa_crewai(pergunta)


def criar_agente_analista_qualidade(llm, usar_tools: bool = True):
    return Agent(
        role="Analista de qualidade laboratorial especializado em farelo de soja.",
        goal=(
            "Avaliar resultados físico-químicos, consultar dados laboratoriais e "
            "usar o modelo preditivo para apoiar decisões de qualidade."
        ),
        backstory=(
            "Você atua em um laboratório físico-químico industrial e interpreta "
            "resultados de farelo de soja, como umidade, proteína, extrato etéreo, "
            "fibras, matéria mineral, urease e solubilidade. Seu objetivo é apoiar "
            "a tomada de decisão, sem substituir critérios oficiais do laboratório."
        ),
        tools=(
            [prever_farelo_soja, consultar_dados_laboratorio, obter_amostra_media]
            if usar_tools
            else []
        ),
        llm=llm,
        verbose=False,
    )


def criar_agente_especialista_modelo(llm, usar_tools: bool = True):
    return Agent(
        role="Especialista em machine learning aplicado a dados laboratoriais.",
        goal=(
            "Explicar o modelo treinado, suas métricas, limitações, variável-alvo, "
            "colunas utilizadas e confiabilidade das previsões."
        ),
        backstory=(
            "Você interpreta o desempenho do modelo FLAML treinado no projeto e "
            "traduz métricas como accuracy, precision, recall, F1, MAE, RMSE e R² "
            "para uma linguagem clara para gestores de laboratório."
        ),
        tools=(
            [obter_metadata_modelo, consultar_dados_laboratorio]
            if usar_tools
            else []
        ),
        llm=llm,
        verbose=False,
    )


def _extrair_valores_previsao(pergunta: str) -> dict[str, float] | None:
    mapa_campos = {
        "umidade_pct": [r"umidade"],
        "proteina_pct": [r"prote[ií]na", r"proteina"],
        "extrato_etereo_pct": [r"extrato\s+et[eé]reo", r"extrato\s+etereo"],
        "fibras_pct": [r"fibras?", r"fibra"],
        "materia_mineral_pct": [r"mat[eé]ria\s+mineral", r"materia\s+mineral"],
        "urease_uph": [r"urease"],
        "solubilidade_pct": [r"solubilidade"],
    }
    valores = {}
    for campo, padroes in mapa_campos.items():
        for padrao in padroes:
            match = re.search(
                rf"{padrao}\s*(?:de|=|:)?\s*(-?\d+(?:[,.]\d+)?)",
                pergunta,
                flags=re.IGNORECASE,
            )
            if match:
                valores[campo] = float(match.group(1).replace(",", "."))
                break

    if set(valores) == set(mapa_campos):
        return valores
    return None


def _gerar_contexto_ferramentas(pergunta: str, roteamento: dict) -> tuple[str, list[str]]:
    blocos = []
    ferramentas = []
    pergunta_lower = pergunta.lower()

    valores_previsao = _extrair_valores_previsao(pergunta)
    if roteamento["usar_analista"] and valores_previsao:
        ferramentas.append("prever_farelo_soja")
        blocos.append(
            "Resultado da API FastAPI para prever_farelo_soja:\n"
            f"{prever_farelo_soja_func(**valores_previsao)}"
        )

    if roteamento["usar_analista"] and "amostra média" in pergunta_lower:
        ferramentas.append("obter_amostra_media")
        blocos.append(
            "Resultado da API FastAPI para obter_amostra_media:\n"
            f"{obter_amostra_media_func()}"
        )

    if (
        roteamento["usar_analista"]
        and (
            "quantas" in pergunta_lower
            or "distribuição" in pergunta_lower
            or "distribuicao" in pergunta_lower
        )
        and "classe" in pergunta_lower
    ):
        ferramentas.append("consultar_dados_laboratorio")
        blocos.append(
            "Resultado da API FastAPI para consultar_dados_laboratorio:\n"
            + consultar_dados_laboratorio_func(
                "SELECT classe_qualidade, COUNT(*) AS quantidade "
                "FROM dataset_lab GROUP BY classe_qualidade "
                "ORDER BY quantidade DESC"
            )
        )

    if (
        roteamento["usar_analista"]
        and ("média" in pergunta_lower or "media" in pergunta_lower)
        and "classe" in pergunta_lower
    ):
        ferramentas.append("consultar_dados_laboratorio")
        blocos.append(
            "Resultado da API FastAPI para consultar_dados_laboratorio:\n"
            + consultar_dados_laboratorio_func(
                "SELECT classe_qualidade, "
                "AVG(umidade_pct) AS media_umidade, "
                "AVG(proteina_pct) AS media_proteina, "
                "AVG(extrato_etereo_pct) AS media_extrato_etereo, "
                "AVG(fibras_pct) AS media_fibras, "
                "AVG(materia_mineral_pct) AS media_materia_mineral, "
                "AVG(urease_uph) AS media_urease, "
                "AVG(solubilidade_pct) AS media_solubilidade "
                "FROM dataset_lab GROUP BY classe_qualidade "
                "ORDER BY classe_qualidade"
            )
        )

    if roteamento["usar_especialista"]:
        ferramentas.append("obter_metadata_modelo")
        blocos.append(
            "Resultado da API FastAPI para obter_metadata_modelo:\n"
            f"{obter_metadata_modelo_func()}"
        )

    if not blocos:
        return "", ferramentas

    return (
        "\n\nContexto já obtido pelas ferramentas Python antes de chamar o LLM "
        "(use estes dados, não invente valores):\n"
        + "\n\n".join(blocos)
    ), ferramentas


def _criar_tarefas(
    pergunta: str,
    roteamento: dict,
    analista,
    especialista,
    contexto_ferramentas: str = "",
) -> list:
    tarefas = []
    contexto_base = (
        f"Pergunta do usuário: {pergunta}\n"
        "Responda em português técnico e claro. Use apenas os dados presentes no "
        "contexto já obtido pelas ferramentas Python. Não afirme que executou "
        "consultas, previsões ou ferramentas além das informações mostradas no "
        "contexto. Nunca invente valores: se faltar algum dado, diga isso de forma "
        "objetiva."
        f"{contexto_ferramentas}"
    )

    if roteamento["usar_analista"]:
        tarefas.append(
            Task(
                description=(
                    f"{contexto_base}\n"
                    "Como analista laboratorial, resolva a parte da pergunta ligada "
                    "à qualidade do farelo de soja com base no contexto fornecido. "
                    "Se houver resultado da API FastAPI, interprete esse resultado. "
                    "Se não houver contexto suficiente, informe a limitação."
                ),
                expected_output=(
                    "Resultado da previsão ou consulta, interpretação laboratorial "
                    "breve e ressalva de que a decisão oficial permanece com o laboratório."
                ),
                agent=analista,
            )
        )

    if roteamento["usar_especialista"]:
        tarefas.append(
            Task(
                description=(
                    f"{contexto_base}\n"
                    "Como especialista em modelo preditivo, explique metadados, "
                    "métricas, variável-alvo, colunas usadas, limitações e "
                    "confiabilidade com base no contexto fornecido. Se não houver "
                    "metadados no contexto, informe a limitação."
                ),
                expected_output=(
                    "Explicação clara do modelo FLAML, métricas disponíveis, alvo, "
                    "colunas usadas e limitações práticas."
                ),
                agent=especialista,
            )
        )

    return tarefas


def executar_crew_lab(pergunta: str) -> dict:
    inicio = time.perf_counter()
    load_dotenv()
    pergunta = (pergunta or "").strip()
    modelo = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL_PADRAO)
    agentes_usados: list[str] = []
    ferramentas_utilizadas: list[str] = []
    erro: str | None = None

    try:
        if not pergunta:
            raise ValueError("Digite uma pergunta para os agentes CrewAI.")

        roteamento = rotear_tarefa_crewai(pergunta)
        agentes_usados = roteamento["agentes"]

        status_ollama = verificar_ollama()
        if status_ollama.get("status") != "ok":
            raise RuntimeError(status_ollama.get("mensagem"))

        status_modelo = modelo_ollama_disponivel(modelo)
        if status_modelo.get("status") != "ok":
            mensagem = status_modelo.get("mensagem") or MENSAGEM_MODELO_INDISPONIVEL
            raise RuntimeError(mensagem)

        if not CREWAI_DISPONIVEL:
            raise RuntimeError(CREWAI_IMPORT_ERROR or "CrewAI não encontrado.")

        if not CREWAI_TOOLS_DISPONIVEL:
            raise RuntimeError(
                CREWAI_TOOLS_IMPORT_ERROR
                or "Ferramentas CrewAI não encontradas."
            )

        contexto_ferramentas, ferramentas_utilizadas = _gerar_contexto_ferramentas(
            pergunta,
            roteamento,
        )
        usar_tools_nativas = False

        llm = _criar_llm_ollama()
        analista = criar_agente_analista_qualidade(
            llm,
            usar_tools=usar_tools_nativas,
        )
        especialista = criar_agente_especialista_modelo(
            llm,
            usar_tools=usar_tools_nativas,
        )
        tarefas = _criar_tarefas(
            pergunta,
            roteamento,
            analista,
            especialista,
            contexto_ferramentas=contexto_ferramentas,
        )

        crew = Crew(
            agents=[analista, especialista],
            tasks=tarefas,
            process=Process.sequential,
            verbose=False,
        )
        resposta = crew.kickoff()

        resultado = {
            "pergunta": pergunta,
            "resposta": str(resposta),
            "agentes": agentes_usados,
            "ferramenta_utilizada": ferramentas_utilizadas,
            "modelo_ollama": modelo,
            "status": "ok",
            "modo_ferramentas": (
                "contexto_precarregado"
                if not usar_tools_nativas
                else "tools_nativas"
            ),
        }
    except Exception as exc:
        erro = str(exc)
        resultado = {
            "pergunta": pergunta,
            "resposta": erro,
            "agentes": agentes_usados,
            "ferramenta_utilizada": ferramentas_utilizadas,
            "modelo_ollama": modelo,
            "status": "erro",
            "traceback": traceback.format_exc(),
        }

    tempo_execucao_ms = int((time.perf_counter() - inicio) * 1000)
    _registrar_trace(
        {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "pergunta": pergunta,
            "resposta": resultado.get("resposta"),
            "status": resultado.get("status"),
            "agentes": resultado.get("agentes", agentes_usados),
            "ferramenta_utilizada": resultado.get(
                "ferramenta_utilizada",
                ferramentas_utilizadas,
            ),
            "modelo_ollama": resultado.get("modelo_ollama", modelo),
            "tempo_execucao_ms": tempo_execucao_ms,
            "erro": erro,
        }
    )
    resultado["tempo_execucao_ms"] = tempo_execucao_ms
    return resultado

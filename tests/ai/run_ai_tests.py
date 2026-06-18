import argparse
import csv
import json
import multiprocessing
import os
import queue
import re
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import REPORTS_DIR, criar_pastas

GOLDEN_DATASET_PATH = PROJECT_ROOT / "tests" / "ai" / "golden_dataset.json"
CONFIG_TESTES_PATH = PROJECT_ROOT / "tests" / "ai" / "config_testes_ia.json"
RESULTADOS_PATH = REPORTS_DIR / "resultados_testes_ia.csv"

THRESHOLDS = {
    "relevancia": 0.70,
    "fidelidade": 0.70,
    "adequacao_tecnica": 0.75,
}
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class TimeoutPerguntaError(Exception):
    pass


class AvaliadorNaoConfiguradoError(Exception):
    def __init__(
        self,
        mensagem: str,
        resposta_bruta_avaliador: str = "",
        erro_avaliador: str = "",
    ) -> None:
        super().__init__(mensagem)
        self.resposta_bruta_avaliador = resposta_bruta_avaliador
        self.erro_avaliador = erro_avaliador or mensagem


class AvaliadorTimeoutError(Exception):
    pass


class RespostaVaziaError(Exception):
    pass


class GeracaoRespostaError(Exception):
    pass


def main() -> None:
    """Executa o golden dataset, avalia respostas e persiste um CSV auditavel."""
    args = parse_args()
    criar_pastas()
    config_testes = carregar_config_testes(CONFIG_TESTES_PATH)

    if args.fast:
        modelo_teste = config_testes.get("modelo_teste_ollama", "qwen2.5:1.5b")
        os.environ["OLLAMA_TEST_MODEL"] = modelo_teste
        os.environ["OLLAMA_MODEL"] = modelo_teste
    if args.judge == "ollama" and not args.judge_model:
        args.judge_model = config_testes.get("modelo_julgador_ollama", "llama3.2:3b")

    imprimir_configuracao_avaliacao(args)

    casos = carregar_golden_dataset(GOLDEN_DATASET_PATH)
    resultados = []

    for caso in casos:
        resultado = executar_caso(caso, args)
        resultados.append(resultado)
        imprimir_linha_execucao(resultado)

    salvar_resultados(resultados, RESULTADOS_PATH)
    imprimir_resumo(resultados)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa testes automatizados de IA.")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Usa resposta direta rapida com Ollama qwen2.5:1.5b, sem CrewAI.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("AI_TEST_TIMEOUT_SECONDS", "75")),
        help="Timeout por pergunta em segundos.",
    )
    parser.add_argument(
        "--allow-heuristic-fallback",
        action="store_true",
        help="Permite fallback heuristico quando DeepEval/OPENAI_API_KEY nao estiver configurado.",
    )
    parser.add_argument(
        "--heuristic",
        action="store_true",
        help="Atalho compativel para avaliar com heuristica local.",
    )
    parser.add_argument(
        "--judge",
        choices=["deepeval", "ollama", "heuristic"],
        default="deepeval",
        help="Avaliador usado nos testes.",
    )
    parser.add_argument(
        "--judge-model",
        default=None,
        help="Modelo Ollama usado como julgador quando --judge ollama.",
    )
    parser.add_argument(
        "--judge-timeout",
        type=int,
        default=int(os.getenv("AI_JUDGE_TIMEOUT_SECONDS", "60")),
        help="Timeout do avaliador Ollama em segundos.",
    )
    args = parser.parse_args()
    if args.heuristic:
        args.judge = "heuristic"
        args.allow_heuristic_fallback = True
    return args


def carregar_golden_dataset(caminho: Path) -> list[dict[str, Any]]:
    with caminho.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def carregar_config_testes(caminho: Path) -> dict[str, Any]:
    if not caminho.exists():
        return {"modelo_teste_ollama": "qwen2.5:1.5b"}
    with caminho.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def imprimir_configuracao_avaliacao(args: argparse.Namespace) -> None:
    if args.judge == "ollama":
        print("Modo de avaliacao: OLLAMA LOCAL")
        print(f"Modelo julgador usado: {args.judge_model}")
    elif args.judge == "heuristic":
        print("Modo de avaliacao: HEURISTICA LOCAL")
        print("Modelo julgador usado: nenhum")
    else:
        print("Modo de avaliacao: DEEPEVAL")
        print("Modelo julgador usado: configuracao DeepEval")


def nome_avaliador(args: argparse.Namespace) -> str:
    if args.judge == "ollama":
        return "OLLAMA LOCAL"
    if args.judge == "heuristic":
        return "HEURISTICA LOCAL"
    return "DEEPEVAL"


def executar_caso(caso: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Executa um caso isolado para impedir que uma falha interrompa toda a suite."""
    data_execucao = datetime.now().isoformat(timespec="seconds")
    pergunta = caso["pergunta"]
    inicio = time.perf_counter()
    resposta_obtida = ""
    contexto_obtido = ""
    contexto_usado_resumido = ""
    resposta_bruta_avaliador = ""
    erro_avaliador = ""
    modelo_resposta = ""
    scores: dict[str, Any] = {
        "score_relevancia": "",
        "score_fidelidade": "",
        "score_adequacao_tecnica": "",
    }
    justificativa = ""
    avaliador_usado = nome_avaliador(args)
    modelo_julgador = args.judge_model if args.judge == "ollama" else ""

    try:
        retorno = chamar_responder(pergunta, args)
        resposta_raw = retorno.get("resposta")
        resposta_obtida = "" if resposta_raw is None else str(resposta_raw)
        contexto_obtido = str(retorno.get("contexto", ""))
        contexto_usado_resumido = resumir_texto(
            "\n".join(
                parte
                for parte in [caso.get("contexto_esperado", ""), contexto_obtido]
                if parte
            ),
            1500,
        )
        modelo_resposta = str(retorno.get("modelo_teste") or retorno.get("modelo") or "")

        if not resposta_obtida.strip():
            tempo_execucao = time.perf_counter() - inicio
            status_final = "ERRO_RESPOSTA_VAZIA"
            justificativa = "Resposta obtida vazia; avaliador nao foi chamado."
            raise RespostaVaziaError

        if contem_erro_interno_resposta(resposta_obtida):
            tempo_execucao = time.perf_counter() - inicio
            status_final = "ERRO_GERACAO_RESPOSTA"
            justificativa = "Resposta obtida contem erro interno da aplicacao; avaliador nao foi chamado."
            raise GeracaoRespostaError

        avaliacao = avaliar_resposta(
            pergunta=pergunta,
            resposta_esperada=caso.get("resposta_esperada", ""),
            resposta_obtida=resposta_obtida,
            contexto_esperado=caso.get("contexto_esperado", ""),
            contexto_obtido=contexto_obtido,
            criterio_avaliacao=caso.get("criterio_avaliacao", ""),
            categoria=caso.get("categoria", ""),
            permitir_fallback=args.allow_heuristic_fallback,
            args=args,
        )
        scores = {
            "score_relevancia": round(avaliacao["score_relevancia"], 4),
            "score_fidelidade": round(avaliacao["score_fidelidade"], 4),
            "score_adequacao_tecnica": round(avaliacao["score_adequacao_tecnica"], 4),
        }
        justificativa = avaliacao["justificativa"]
        status_final = definir_status_qualidade(avaliacao)
        tempo_execucao = time.perf_counter() - inicio
    except TimeoutPerguntaError:
        tempo_execucao = time.perf_counter() - inicio
        resposta_obtida = "ERRO_TIMEOUT"
        status_final = "ERRO_TIMEOUT"
        justificativa = f"Timeout apos {args.timeout} segundos."
    except AvaliadorTimeoutError as erro:
        tempo_execucao = time.perf_counter() - inicio
        status_final = "ERRO_TIMEOUT"
        justificativa = str(erro)
    except AvaliadorNaoConfiguradoError as erro:
        tempo_execucao = time.perf_counter() - inicio
        status_final = "ERRO_AVALIADOR"
        justificativa = str(erro)
        resposta_bruta_avaliador = erro.resposta_bruta_avaliador
        erro_avaliador = erro.erro_avaliador
    except RespostaVaziaError:
        pass
    except GeracaoRespostaError:
        pass
    except Exception as erro:
        tempo_execucao = time.perf_counter() - inicio
        resposta_obtida = f"ERRO: {type(erro).__name__}: {erro}"
        status_final = "ERRO_GERACAO_RESPOSTA"
        justificativa = f"Falha operacional ao gerar resposta: {erro}"

    return {
        "id": caso.get("id", ""),
        "pergunta": pergunta,
        "resposta_esperada": caso.get("resposta_esperada", ""),
        "resposta_obtida": resposta_obtida,
        "categoria": caso.get("categoria", ""),
        **scores,
        "status_final": status_final,
        "justificativa": justificativa,
        "data_execucao": data_execucao,
        "tempo_execucao_segundos": round(tempo_execucao, 3),
        "avaliador_usado": avaliador_usado,
        "modelo_julgador": modelo_julgador,
        "modelo_resposta": modelo_resposta,
        "contexto_usado_resumido": contexto_usado_resumido,
        "resposta_bruta_avaliador": resposta_bruta_avaliador,
        "erro_avaliador": erro_avaliador,
    }


def chamar_responder(pergunta: str, args: argparse.Namespace) -> dict[str, Any]:
    if args.fast:
        from ai.agentes.ai_responder import responder_pergunta_teste

        return responder_pergunta_teste(pergunta)
    return chamar_responder_com_timeout(pergunta, args.timeout)


def chamar_responder_com_timeout(pergunta: str, timeout_segundos: int) -> dict[str, Any]:
    contexto_mp = multiprocessing.get_context("spawn")
    fila = contexto_mp.Queue()
    processo = contexto_mp.Process(
        target=_worker_responder_pergunta,
        args=(pergunta, fila),
    )
    processo.start()
    processo.join(timeout_segundos)

    if processo.is_alive():
        processo.terminate()
        processo.join(5)
        raise TimeoutPerguntaError()

    try:
        payload = fila.get(timeout=5)
    except queue.Empty as exc:
        raise RuntimeError("Processo de resposta terminou sem retornar resultado.") from exc
    finally:
        fila.close()
        fila.join_thread()

    if payload.get("status") == "erro":
        raise RuntimeError(payload.get("erro", "Erro desconhecido no responder."))
    return payload["resultado"]


def _worker_responder_pergunta(pergunta: str, fila) -> None:
    try:
        from ai.agentes.ai_responder import responder_pergunta

        fila.put({"status": "ok", "resultado": responder_pergunta(pergunta)})
    except Exception as erro:
        fila.put({"status": "erro", "erro": f"{type(erro).__name__}: {erro}"})


def avaliar_resposta(
    pergunta: str,
    resposta_esperada: str,
    resposta_obtida: str,
    contexto_esperado: str,
    contexto_obtido: str,
    criterio_avaliacao: str,
    categoria: str,
    permitir_fallback: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    contexto_completo = "\n".join(
        parte for parte in [contexto_esperado, contexto_obtido] if parte
    )
    if args.judge == "ollama":
        return avaliar_com_ollama(
            pergunta=pergunta,
            resposta_esperada=resposta_esperada,
            resposta_obtida=resposta_obtida,
            contexto=contexto_completo,
            categoria=categoria,
            modelo_julgador=args.judge_model,
            timeout_segundos=args.judge_timeout,
        )

    if args.judge == "heuristic":
        relevancia = similaridade_lexical(pergunta + " " + resposta_esperada, resposta_obtida)
        fidelidade = similaridade_lexical(contexto_completo, resposta_obtida) if contexto_completo else 0.0
        adequacao, justificativa_adequacao = avaliar_adequacao_tecnica_heuristica(
            pergunta,
            resposta_obtida,
            resposta_esperada,
            criterio_avaliacao,
        )
        return {
            "score_relevancia": relevancia,
            "score_fidelidade": fidelidade,
            "score_adequacao_tecnica": adequacao,
            "justificativa": (
                "Avaliacao heuristica local. "
                f"{justificativa_adequacao}"
            ),
        }

    relevancia, justificativa_relevancia = avaliar_relevancia(
        pergunta,
        resposta_obtida,
        resposta_esperada,
        permitir_fallback,
    )
    fidelidade, justificativa_fidelidade = avaliar_fidelidade(
        pergunta,
        resposta_obtida,
        contexto_completo,
        permitir_fallback,
    )
    adequacao, justificativa_adequacao = avaliar_adequacao_tecnica_laboratorial(
        pergunta=pergunta,
        resposta=resposta_obtida,
        resposta_esperada=resposta_esperada,
        criterio_avaliacao=criterio_avaliacao,
        permitir_fallback=permitir_fallback,
    )

    return {
        "score_relevancia": relevancia,
        "score_fidelidade": fidelidade,
        "score_adequacao_tecnica": adequacao,
        "justificativa": " | ".join(
            [justificativa_relevancia, justificativa_fidelidade, justificativa_adequacao]
        ),
    }


def avaliar_com_ollama(
    pergunta: str,
    resposta_esperada: str,
    resposta_obtida: str,
    contexto: str,
    categoria: str,
    modelo_julgador: str = "llama3.2:3b",
    timeout_segundos: int = 45,
) -> dict[str, Any]:
    prompt = montar_prompt_julgador(
        pergunta=pergunta,
        resposta_esperada=resposta_esperada,
        resposta_obtida=resposta_obtida,
        contexto=contexto,
        categoria=categoria,
    )
    try:
        resposta = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": modelo_julgador,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "keep_alive": "10m",
                "options": {
                    "temperature": 0.0,
                    "num_predict": 220,
                    "num_ctx": 1024,
                },
            },
            timeout=timeout_segundos,
        )
        resposta.raise_for_status()
    except requests.Timeout as erro:
        raise AvaliadorTimeoutError(
            f"Timeout do avaliador Ollama apos {timeout_segundos} segundos."
        ) from erro
    except Exception as erro:
        raise AvaliadorNaoConfiguradoError(
            f"Falha no avaliador Ollama: {erro}",
            erro_avaliador=f"Falha no avaliador Ollama: {erro}",
        ) from erro

    try:
        saida_bruta = str(resposta.json().get("response", "")).strip()
    except ValueError as erro:
        texto_bruto = resposta.text[:1000]
        raise AvaliadorNaoConfiguradoError(
            "Resposta HTTP do Ollama nao estava em JSON valido.",
            resposta_bruta_avaliador=texto_bruto,
            erro_avaliador=f"Resposta HTTP invalida: {erro}",
        ) from erro
    dados = extrair_json_avaliador(saida_bruta)
    if dados is None:
        saida_curta = saida_bruta[:500].replace("\n", " ")
        raise AvaliadorNaoConfiguradoError(
            f"JSON invalido retornado pelo Ollama. Saida bruta: {saida_curta}",
            resposta_bruta_avaliador=saida_bruta,
            erro_avaliador="JSON invalido retornado pelo Ollama.",
        )

    try:
        return {
            "score_relevancia": normalizar_score(
                obter_campo_avaliador(dados, ["score_relevancia", "score_relevância", "relevancia", "relevância"])
            ),
            "score_fidelidade": normalizar_score(
                obter_campo_avaliador(dados, ["score_fidelidade", "fidelidade", "faithfulness"])
            ),
            "score_adequacao_tecnica": normalizar_score(
                obter_campo_avaliador(
                    dados,
                    [
                        "score_adequacao_tecnica",
                        "score_adequação_técnica",
                        "adequacao_tecnica",
                        "adequação_técnica",
                        "adequacao",
                        "adequação",
                    ],
                )
            ),
            "justificativa": str(dados.get("justificativa", dados.get("reason", ""))).strip()[:800],
        }
    except (KeyError, TypeError, ValueError) as erro:
        saida_curta = saida_bruta[:500].replace("\n", " ")
        raise AvaliadorNaoConfiguradoError(
            f"JSON do avaliador com score ausente ou invalido: {erro}. Saida bruta: {saida_curta}",
            resposta_bruta_avaliador=saida_bruta,
            erro_avaliador=f"JSON com score ausente ou invalido: {erro}",
        ) from erro


def obter_campo_avaliador(dados: dict[str, Any], nomes: list[str]) -> Any:
    for nome in nomes:
        if nome in dados:
            return dados[nome]
    raise KeyError(nomes[0])


def montar_prompt_julgador(
    pergunta: str,
    resposta_esperada: str,
    resposta_obtida: str,
    contexto: str,
    categoria: str,
) -> str:
    contexto_reduzido = sanitizar_texto_para_prompt(resumir_texto(contexto or "Nao informado.", 1500))
    resposta_esperada_reduzida = sanitizar_texto_para_prompt(resumir_texto(resposta_esperada or "", 800))
    resposta_obtida_reduzida = sanitizar_texto_para_prompt(resumir_texto(resposta_obtida or "", 1200))
    return f"""
RESPONDA SOMENTE COM JSON VALIDO.
NAO escreva explicacoes fora do JSON.
NAO use markdown.
NAO use ```json.
NAO inclua texto antes ou depois.

Voce e um juiz de respostas de IA para laboratorio analitico.
Avalie a resposta obtida. Nao responda a pergunta. Nao repita o contexto.

Formato obrigatorio:
{{
  "score_relevancia": 0.0,
  "score_fidelidade": 0.0,
  "score_adequacao_tecnica": 0.0,
  "justificativa": "maximo 6 palavras"
}}

Criterios:
- Relevancia: responde diretamente, sem fugir do tema e sem genericidade.
- Fidelidade: apoia-se no contexto, nao inventa colunas, metricas, resultados ou valores; se faltam dados, informa limite.
- Adequacao tecnica: linguagem laboratorial; diferencia previsao, calculo e interpretacao quando aplicavel; nao trata previsao como certeza; nao recomenda liberacao automatica; recomenda analista/responsavel quando necessario.

Categoria: {categoria}
Pergunta: {pergunta}
Resposta esperada: {resposta_esperada_reduzida}
Resposta obtida: {resposta_obtida_reduzida}
Contexto usado: {contexto_reduzido}

Regras finais:
- A justificativa deve ter no maximo 6 palavras.
- Use numeros entre 0.0 e 1.0.
- Feche o JSON com }}.
""".strip()


def sanitizar_texto_para_prompt(texto: str) -> str:
    return (
        (texto or "")
        .replace("{", "(")
        .replace("}", ")")
        .replace("[", "(")
        .replace("]", ")")
    )


def resumir_texto(texto: str, limite: int) -> str:
    texto = re.sub(r"\s+", " ", texto or "").strip()
    if len(texto) <= limite:
        return texto
    return texto[:limite].rstrip() + "..."


def extrair_json_avaliador(saida: str) -> dict[str, Any] | None:
    for candidato in candidatos_json_avaliador(saida):
        try:
            return json.loads(candidato)
        except json.JSONDecodeError:
            continue
    return None


def candidatos_json_avaliador(saida: str) -> list[str]:
    sem_markdown = remover_markdown_json(saida)
    candidatos = [saida, extrair_primeiro_bloco_json(saida), sem_markdown, extrair_primeiro_bloco_json(sem_markdown)]
    return [candidato.strip() for candidato in candidatos if candidato and candidato.strip()]


def extrair_primeiro_bloco_json(texto: str) -> str:
    inicio = texto.find("{")
    if inicio == -1:
        return ""

    profundidade = 0
    for indice, caractere in enumerate(texto[inicio:], start=inicio):
        if caractere == "{":
            profundidade += 1
        elif caractere == "}":
            profundidade -= 1
            if profundidade == 0:
                return texto[inicio : indice + 1]
    return ""


def remover_markdown_json(texto: str) -> str:
    texto = (texto or "").strip()
    texto = re.sub(r"^```(?:json)?\s*", "", texto, flags=re.IGNORECASE)
    texto = re.sub(r"\s*```$", "", texto)
    return texto.strip()


def avaliar_relevancia(
    pergunta: str,
    resposta_obtida: str,
    resposta_esperada: str,
    permitir_fallback: bool,
) -> tuple[float, str]:
    if not os.getenv("OPENAI_API_KEY"):
        if not permitir_fallback:
            raise AvaliadorNaoConfiguradoError(
                "DeepEval nao configurado: OPENAI_API_KEY ausente."
            )
        score = similaridade_lexical(pergunta + " " + resposta_esperada, resposta_obtida)
        return score, "Fallback heuristico de relevancia habilitado explicitamente."

    try:
        from deepeval.metrics import AnswerRelevancyMetric
        from deepeval.test_case import LLMTestCase

        caso = LLMTestCase(input=pergunta, actual_output=resposta_obtida)
        metrica = AnswerRelevancyMetric(threshold=THRESHOLDS["relevancia"])
        metrica.measure(caso)
        return normalizar_score(metrica.score), f"DeepEval Answer Relevancy: {metrica.reason}"
    except Exception as erro:
        if not permitir_fallback:
            raise AvaliadorNaoConfiguradoError(f"Falha no DeepEval Answer Relevancy: {erro}") from erro
        score = similaridade_lexical(pergunta + " " + resposta_esperada, resposta_obtida)
        return score, f"Fallback heuristico de relevancia habilitado: {erro}"


def avaliar_fidelidade(
    pergunta: str,
    resposta_obtida: str,
    contexto: str,
    permitir_fallback: bool,
) -> tuple[float, str]:
    if not os.getenv("OPENAI_API_KEY"):
        if not permitir_fallback:
            raise AvaliadorNaoConfiguradoError(
                "DeepEval nao configurado: OPENAI_API_KEY ausente."
            )
        score = similaridade_lexical(contexto, resposta_obtida) if contexto else 0.0
        return score, "Fallback heuristico de fidelidade habilitado explicitamente."

    try:
        from deepeval.metrics import FaithfulnessMetric
        from deepeval.test_case import LLMTestCase

        caso = LLMTestCase(
            input=pergunta,
            actual_output=resposta_obtida,
            retrieval_context=[contexto] if contexto else [],
        )
        metrica = FaithfulnessMetric(threshold=THRESHOLDS["fidelidade"])
        metrica.measure(caso)
        return normalizar_score(metrica.score), f"DeepEval Faithfulness: {metrica.reason}"
    except Exception as erro:
        if not permitir_fallback:
            raise AvaliadorNaoConfiguradoError(f"Falha no DeepEval Faithfulness: {erro}") from erro
        score = similaridade_lexical(contexto, resposta_obtida) if contexto else 0.0
        return score, f"Fallback heuristico de fidelidade habilitado: {erro}"


def avaliar_adequacao_tecnica_laboratorial(
    pergunta: str,
    resposta: str,
    resposta_esperada: str,
    criterio_avaliacao: str,
    permitir_fallback: bool,
) -> tuple[float, str]:
    if not os.getenv("OPENAI_API_KEY"):
        if not permitir_fallback:
            raise AvaliadorNaoConfiguradoError(
                "DeepEval nao configurado: OPENAI_API_KEY ausente."
            )
        score, justificativa = avaliar_adequacao_tecnica_heuristica(
            pergunta,
            resposta,
            resposta_esperada,
            criterio_avaliacao,
        )
        return score, justificativa + "; fallback heuristico habilitado explicitamente."

    try:
        from deepeval.metrics import GEval
        from deepeval.test_case import LLMTestCase, LLMTestCaseParams

        caso = LLMTestCase(
            input=pergunta,
            actual_output=resposta,
            expected_output=resposta_esperada,
        )
        metrica = GEval(
            name="AdequacaoTecnicaLaboratorial",
            criteria=(
                "Avalie se a resposta nao inventa dados, respeita limites do modelo, "
                "diferencia previsao, calculo e interpretacao quando aplicavel, nao "
                "recomenda liberacao automatica de amostras e usa linguagem tecnica "
                "adequada para laboratorio."
            ),
            evaluation_params=[
                LLMTestCaseParams.INPUT,
                LLMTestCaseParams.ACTUAL_OUTPUT,
                LLMTestCaseParams.EXPECTED_OUTPUT,
            ],
            threshold=THRESHOLDS["adequacao_tecnica"],
        )
        metrica.measure(caso)
        return normalizar_score(metrica.score), f"DeepEval AdequacaoTecnicaLaboratorial: {metrica.reason}"
    except Exception as erro:
        if not permitir_fallback:
            raise AvaliadorNaoConfiguradoError(f"Falha no DeepEval GEval: {erro}") from erro
        score, justificativa = avaliar_adequacao_tecnica_heuristica(
            pergunta,
            resposta,
            resposta_esperada,
            criterio_avaliacao,
        )
        return score, f"{justificativa}; fallback heuristico habilitado: {erro}"


def avaliar_adequacao_tecnica_heuristica(
    pergunta: str,
    resposta: str,
    resposta_esperada: str,
    criterio_avaliacao: str,
) -> tuple[float, str]:
    texto = normalizar_texto(f"{pergunta} {resposta}")
    score = 1.0
    motivos = []

    if contem_algum(
        texto,
        [
            "liberar automaticamente",
            "liberacao automatica",
            "pode liberar o lote",
            "aprovado automaticamente",
            "dispensa analise",
        ],
    ):
        score -= 0.35
        motivos.append("sugere liberacao automatica")

    if contem_algum(texto, ["garante", "sempre", "nunca falha", "100%", "sem erro"]):
        score -= 0.15
        motivos.append("usa afirmacao absoluta sobre o modelo")

    if pergunta_exige_limite_modelo(pergunta) and not contem_algum(
        texto,
        ["limite", "limitacao", "apoio", "validacao", "criterio", "analista"],
    ):
        score -= 0.20
        motivos.append("nao explicita limites do modelo")

    if pergunta_exige_diferenciacao(pergunta) and not contem_algum(
        texto,
        ["previsao", "calculo", "interpretacao"],
    ):
        score -= 0.20
        motivos.append("nao diferencia previsao, calculo e interpretacao")

    if not contem_algum(
        texto,
        [
            "modelo",
            "metrica",
            "amostra",
            "laboratorio",
            "farelo",
            "soja",
            "umidade",
            "proteina",
            "urease",
            "solubilidade",
            "analista",
            "validacao",
        ],
    ):
        score -= 0.15
        motivos.append("linguagem tecnica insuficiente")

    alinhamento = similaridade_lexical(
        resposta_esperada + " " + criterio_avaliacao,
        resposta,
    )
    score = min(score, 0.65 + (0.35 * alinhamento))
    score = max(0.0, min(1.0, score))

    if not motivos:
        motivos.append("criterios tecnicos atendidos pela avaliacao heuristica")
    return score, "AdequacaoTecnicaLaboratorial: " + "; ".join(motivos)


def definir_status_qualidade(avaliacao: dict[str, float]) -> str:
    if (
        avaliacao["score_relevancia"] >= THRESHOLDS["relevancia"]
        and avaliacao["score_fidelidade"] >= THRESHOLDS["fidelidade"]
        and avaliacao["score_adequacao_tecnica"] >= THRESHOLDS["adequacao_tecnica"]
    ):
        return "APROVADO"
    return "REPROVADO"


def salvar_resultados(resultados: list[dict[str, Any]], caminho: Path) -> None:
    """Salva resultados com colunas estaveis para comparacao entre execucoes."""
    caminho.parent.mkdir(parents=True, exist_ok=True)
    campos = [
        "id",
        "pergunta",
        "resposta_esperada",
        "resposta_obtida",
        "categoria",
        "score_relevancia",
        "score_fidelidade",
        "score_adequacao_tecnica",
        "status_final",
        "justificativa",
        "data_execucao",
        "tempo_execucao_segundos",
        "avaliador_usado",
        "modelo_julgador",
        "modelo_resposta",
        "contexto_usado_resumido",
        "resposta_bruta_avaliador",
        "erro_avaliador",
    ]
    with caminho.open("w", newline="", encoding="utf-8-sig") as arquivo:
        writer = csv.DictWriter(arquivo, fieldnames=campos)
        writer.writeheader()
        writer.writerows(resultados)


def imprimir_linha_execucao(resultado: dict[str, Any]) -> None:
    scores = (
        f"rel={resultado.get('score_relevancia', '')}, "
        f"fid={resultado.get('score_fidelidade', '')}, "
        f"tec={resultado.get('score_adequacao_tecnica', '')}"
    )
    print(
        f"{resultado['id']} | {resultado['categoria']} | "
        f"{resultado['tempo_execucao_segundos']:.3f}s | "
        f"{resultado['status_final']} | {scores} | {resultado['pergunta']}"
        ,
        flush=True,
    )
    if resultado.get("status_final") == "ERRO_AVALIADOR":
        bruto = str(resultado.get("resposta_bruta_avaliador", ""))[:200].replace("\n", " ")
        print(
            f"{resultado['id']} | Falha ao interpretar JSON do avaliador | {bruto}",
            flush=True,
        )


def imprimir_resumo(resultados: list[dict[str, Any]]) -> None:
    total = len(resultados)
    aprovados = contar_status(resultados, "APROVADO")
    reprovados = contar_status(resultados, "REPROVADO")
    timeouts = contar_status(resultados, "ERRO_TIMEOUT")
    erros_avaliador = contar_status(resultados, "ERRO_AVALIADOR")
    erros_geracao = contar_status(resultados, "ERRO_GERACAO_RESPOSTA")
    respostas_vazias = contar_status(resultados, "ERRO_RESPOSTA_VAZIA")
    tempos = [float(item["tempo_execucao_segundos"]) for item in resultados]
    tempo_medio = sum(tempos) / len(tempos) if tempos else 0.0
    mais_lenta = max(resultados, key=lambda item: float(item["tempo_execucao_segundos"]), default=None)

    falhas_por_categoria = Counter(
        item["categoria"] for item in resultados if item["status_final"] == "REPROVADO"
    )

    print("\nResumo dos testes de IA")
    print(f"Total de testes: {total}")
    print(f"Aprovados: {aprovados}")
    print(f"Reprovados por qualidade: {reprovados}")
    print(f"Erros por timeout: {timeouts}")
    print(f"Erros do avaliador: {erros_avaliador}")
    print(f"Erros de geracao da resposta: {erros_geracao}")
    print(f"Respostas vazias: {respostas_vazias}")
    print(f"Tempo medio por pergunta: {tempo_medio:.3f}s")
    if mais_lenta:
        print(
            "Pergunta mais lenta: "
            f"{mais_lenta['id']} ({mais_lenta['tempo_execucao_segundos']:.3f}s) - "
            f"{mais_lenta['pergunta']}"
        )
    print(f"CSV gerado em: {RESULTADOS_PATH}")
    print("Categorias com falhas reais de qualidade:")
    if falhas_por_categoria:
        for categoria, quantidade in falhas_por_categoria.most_common():
            print(f"- {categoria}: {quantidade}")
    else:
        print("- nenhuma")


def contar_status(resultados: list[dict[str, Any]], status: str) -> int:
    return sum(1 for item in resultados if item["status_final"] == status)


def normalizar_score(score: Any) -> float:
    try:
        valor = float(score)
    except (TypeError, ValueError):
        raise ValueError(f"Score invalido: {score!r}")
    return max(0.0, min(1.0, valor))


def similaridade_lexical(referencia: str, resposta: str) -> float:
    termos_referencia = set(tokenizar(referencia))
    termos_resposta = set(tokenizar(resposta))
    if not termos_referencia or not termos_resposta:
        return 0.0
    return len(termos_referencia.intersection(termos_resposta)) / len(termos_referencia)


def tokenizar(texto: str) -> list[str]:
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
        "na",
        "no",
        "o",
        "os",
        "para",
        "por",
        "que",
        "se",
        "um",
        "uma",
    }
    return [
        termo
        for termo in re.findall(r"[a-z0-9_]{3,}", normalizar_texto(texto))
        if termo not in stopwords
    ]


def normalizar_texto(texto: str) -> str:
    substituicoes = str.maketrans(
        "áàâãäéèêëíìîïóòôõöúùûüç",
        "aaaaaeeeeiiiiooooouuuuc",
    )
    return (texto or "").lower().translate(substituicoes)


def contem_algum(texto: str, termos: list[str]) -> bool:
    texto_normalizado = normalizar_texto(texto)
    return any(normalizar_texto(termo) in texto_normalizado for termo in termos)


def contem_erro_interno_resposta(resposta: str) -> bool:
    texto = normalizar_texto(resposta)
    padroes_erro = [
        "erro:",
        "erro_timeout",
        "traceback",
        "exception",
        "runtimeerror",
        "valueerror",
        "falha no ollama local",
        "falha operacional",
        "falha ao executar",
        "falha ao gerar",
        "api local nao encontrada",
        "connection refused",
        "read timed out",
        "max retries exceeded",
    ]
    return any(padrao in texto for padrao in padroes_erro)


def pergunta_exige_limite_modelo(pergunta: str) -> bool:
    return contem_algum(pergunta, ["modelo", "predicao", "previsao", "flaml", "liberar"])


def pergunta_exige_diferenciacao(pergunta: str) -> bool:
    return contem_algum(pergunta, ["diferenca", "previsao", "calculo", "interpretacao"])


if __name__ == "__main__":
    main()

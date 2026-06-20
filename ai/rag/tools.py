import json
import logging
import traceback
from typing import Any, Type

from pydantic import BaseModel, Field

from ai.rag.index import MENSAGEM_RAG_INDISPONIVEL, verificar_disponibilidade_rag
from ai.rag.retriever import buscar_trechos, formatar_trechos_para_contexto


LOGGER = logging.getLogger(__name__)

class ConsultaBaseConhecimentoInput(BaseModel):
    pergunta: str = Field(..., description="Pergunta para busca semantica na base RAG.")


def tool_consultar_base_conhecimento(pergunta: str, k: int = 4) -> str:
    status_rag = verificar_disponibilidade_rag()
    if not status_rag["disponivel"]:
        return json.dumps(
            {
                "erro": status_rag["motivo"],
                "mensagem": MENSAGEM_RAG_INDISPONIVEL,
            },
            ensure_ascii=False,
        )
    try:
        trechos = buscar_trechos(pergunta, k=k)
        if not trechos:
            return (
                "Base de conhecimento consultada, mas nenhum trecho relevante foi "
                "encontrado. Informe ao usuario que nao ha informacao suficiente nos "
                "documentos indexados e evite inventar detalhes."
            )

        contexto = formatar_trechos_para_contexto(trechos)
        return (
            "Use os trechos abaixo como contexto documental da base RAG. "
            "Responda apenas quando os trechos sustentarem a resposta; se faltarem "
            "dados, diga claramente que a base nao contem informacao suficiente.\n\n"
            f"{contexto}"
        )
    except BaseException as erro:
        if isinstance(erro, (KeyboardInterrupt, SystemExit)):
            raise
        LOGGER.error(
            "Falha técnica ao consultar o RAG. tipo=%s detalhe=%s\n%s",
            type(erro).__name__,
            erro,
            traceback.format_exc(),
        )
        return json.dumps(
            {
                "erro": "Base de conhecimento temporariamente indisponível.",
                "orientacao": (
                    "Verifique se o ChromaDB esta instalado com "
                    "python -m pip install chromadb, se o Ollama esta ativo e se o "
                    "modelo nomic-embed-text foi baixado com "
                    "ollama pull nomic-embed-text."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )


try:
    from crewai.tools import BaseTool

    CREWAI_RAG_TOOL_DISPONIVEL = True
except Exception:
    BaseTool = object
    CREWAI_RAG_TOOL_DISPONIVEL = False


if CREWAI_RAG_TOOL_DISPONIVEL:

    class ConsultarBaseConhecimentoTool(BaseTool):
        name: str = "tool_consultar_base_conhecimento"
        description: str = (
            "Busca trechos relevantes na base de conhecimento RAG local em ChromaDB "
            "usando embeddings do Ollama. Use antes de responder perguntas tecnicas "
            "sobre documentos do projeto."
        )
        args_schema: Type[BaseModel] = ConsultaBaseConhecimentoInput

        def _run(self, pergunta: str) -> str:
            return tool_consultar_base_conhecimento(pergunta)


    consultar_base_conhecimento = ConsultarBaseConhecimentoTool()
else:
    consultar_base_conhecimento = None

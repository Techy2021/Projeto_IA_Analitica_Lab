import hashlib
import importlib.util
import logging
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

from ai.rag.loader import carregar_documentos, quebrar_em_chunks
from app.config import (
    INACTIVE_DOCUMENTS_DIR,
    KNOWLEDGE_BASE_DIR,
    VECTORSTORE_DIR,
    criar_pastas,
)


LOGGER = logging.getLogger(__name__)

RAG_COLLECTION_NAME = "base_conhecimento_lab"
OLLAMA_EMBED_MODEL_PADRAO = "nomic-embed-text"
MENSAGEM_CHROMADB_INDISPONIVEL = (
    "ChromaDB nao esta instalado. Execute: python -m pip install chromadb"
)
MENSAGEM_RAG_INDISPONIVEL = (
    "Base de conhecimento RAG indisponível. A resposta será gerada sem "
    "recuperação de documentos."
)
MENSAGEM_EMBEDDING_INDISPONIVEL = (
    "Modelo de embedding nao encontrado no Ollama. Execute: "
    "ollama pull nomic-embed-text"
)


def obter_config_embedding() -> tuple[str, str]:
    load_dotenv()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    modelo = os.getenv("OLLAMA_EMBED_MODEL", OLLAMA_EMBED_MODEL_PADRAO)
    return base_url, modelo


def chromadb_disponivel() -> bool:
    """Verifica o pacote sem importá-lo nem inicializar a base vetorial."""
    return importlib.util.find_spec("chromadb") is not None


def verificar_disponibilidade_rag() -> dict[str, Any]:
    """Diagnóstico leve usado antes de qualquer recuperação documental."""
    if not chromadb_disponivel():
        return {
            "disponivel": False,
            "motivo": "chromadb_ausente",
            "mensagem": MENSAGEM_RAG_INDISPONIVEL,
        }
    if not VECTORSTORE_DIR.exists() or not any(VECTORSTORE_DIR.iterdir()):
        return {
            "disponivel": False,
            "motivo": "base_vetorial_ausente",
            "mensagem": MENSAGEM_RAG_INDISPONIVEL,
        }
    try:
        total = obter_collection().count()
    except BaseException as erro:
        if isinstance(erro, (KeyboardInterrupt, SystemExit)):
            raise
        LOGGER.error(
            "Base vetorial RAG inacessível. tipo=%s detalhe=%s",
            type(erro).__name__,
            erro,
        )
        return {
            "disponivel": False,
            "motivo": "base_vetorial_inacessivel",
            "mensagem": MENSAGEM_RAG_INDISPONIVEL,
        }
    if total <= 0:
        return {
            "disponivel": False,
            "motivo": "base_vetorial_vazia",
            "mensagem": MENSAGEM_RAG_INDISPONIVEL,
        }
    return {
        "disponivel": True,
        "motivo": None,
        "mensagem": "Base de conhecimento RAG disponível.",
        "vetores": total,
    }


def obter_collection():
    if not chromadb_disponivel():
        raise RuntimeError(MENSAGEM_CHROMADB_INDISPONIVEL)
    import chromadb

    criar_pastas()
    cliente = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))
    return cliente.get_or_create_collection(name=RAG_COLLECTION_NAME)


def verificar_modelo_embedding(modelo: str | None = None) -> dict[str, Any]:
    base_url, modelo_configurado = obter_config_embedding()
    modelo = modelo or modelo_configurado
    try:
        resposta = requests.get(f"{base_url}/api/tags", timeout=5)
        resposta.raise_for_status()
        modelos = [
            item.get("name") or item.get("model")
            for item in resposta.json().get("models", [])
        ]
        modelos = [str(item) for item in modelos if item]
        disponivel = _modelo_disponivel(modelo, modelos)
        return {
            "status": "ok" if disponivel else "erro",
            "base_url": base_url,
            "modelo": modelo,
            "modelos": modelos,
            "mensagem": (
                f"Modelo de embedding {modelo} disponivel."
                if disponivel
                else MENSAGEM_EMBEDDING_INDISPONIVEL
            ),
        }
    except requests.exceptions.RequestException:
        return {
            "status": "erro",
            "base_url": base_url,
            "modelo": modelo,
            "modelos": [],
            "mensagem": "Ollama nao encontrado. Abra o Ollama ou execute ollama serve.",
        }


def gerar_embedding(texto: str, modelo: str | None = None) -> list[float]:
    base_url, modelo_configurado = obter_config_embedding()
    modelo = modelo or modelo_configurado
    try:
        resposta = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": modelo, "prompt": texto},
            timeout=60,
        )
        resposta.raise_for_status()
        embedding = resposta.json().get("embedding")
        if not embedding:
            raise RuntimeError("Ollama nao retornou embedding para o texto informado.")
        return embedding
    except requests.exceptions.HTTPError as erro:
        detalhe = erro.response.text if erro.response is not None else str(erro)
        raise RuntimeError(
            f"Erro ao gerar embedding com Ollama. Verifique o modelo {modelo}. "
            "Se necessario execute: ollama pull nomic-embed-text. "
            f"Detalhe: {detalhe}"
        ) from erro
    except requests.exceptions.RequestException as erro:
        raise RuntimeError(
            "Nao foi possivel conectar ao Ollama para gerar embeddings. "
            "Abra o Ollama ou execute ollama serve."
        ) from erro


def indexar_documentos(
    pasta: Path = KNOWLEDGE_BASE_DIR,
    tamanho_chunk: int = 250,
    sobreposicao: int = 40,
    modelo: str | None = None,
) -> dict[str, Any]:
    """Sincroniza o indice com todos e somente os documentos ativos."""
    return reconstruir_indice_rag(
        pasta=pasta,
        tamanho_chunk=tamanho_chunk,
        sobreposicao=sobreposicao,
        modelo=modelo,
    )


def reconstruir_indice_rag(
    pasta: Path = KNOWLEDGE_BASE_DIR,
    tamanho_chunk: int = 250,
    sobreposicao: int = 40,
    modelo: str | None = None,
) -> dict[str, Any]:
    """Recria a collection usando apenas os arquivos presentes na pasta ativa.

    Todos os embeddings sao preparados antes de remover o indice atual. Assim,
    uma falha do Ollama durante a preparacao nao destrói a collection existente.
    """
    status_modelo = verificar_modelo_embedding(modelo)
    if status_modelo["status"] != "ok":
        raise RuntimeError(status_modelo["mensagem"])

    documentos = carregar_documentos(pasta)
    ids: list[str] = []
    textos: list[str] = []
    embeddings: list[list[float]] = []
    metadatas: list[dict[str, Any]] = []
    arquivos_indexados = []

    for documento in documentos:
        chunks = quebrar_em_chunks(
            documento.texto,
            tamanho_chunk=tamanho_chunk,
            sobreposicao=sobreposicao,
        )
        if not chunks:
            continue

        for indice, chunk in enumerate(chunks, start=1):
            ids.append(_gerar_id(documento.nome_arquivo, indice, chunk))
            textos.append(chunk)
            embeddings.append(gerar_embedding(chunk, modelo=modelo))
            metadatas.append(
                {
                    "source_file": documento.nome_arquivo,
                    "source_path": documento.caminho,
                    "chunk": indice,
                }
            )

        arquivos_indexados.append(
            {"arquivo": documento.nome_arquivo, "chunks": len(chunks)}
        )

    collection = obter_collection()
    registros_anteriores = collection.get(include=[])
    ids_anteriores = registros_anteriores.get("ids") or []
    if ids_anteriores:
        collection.delete(ids=ids_anteriores)

    if ids:
        collection.add(
            ids=ids,
            documents=textos,
            metadatas=metadatas,
            embeddings=embeddings,
        )

    return {
        "arquivos": len(arquivos_indexados),
        "chunks": len(ids),
        "detalhes": arquivos_indexados,
        "collection": RAG_COLLECTION_NAME,
        "vectorstore": str(VECTORSTORE_DIR),
        "modelo_embedding": status_modelo["modelo"],
    }


def contar_documentos_indexados() -> int:
    return obter_collection().count()


def listar_documentos_rag() -> list[dict[str, Any]]:
    """Lista fontes realmente presentes na collection e suas quantidades."""
    collection = obter_collection()
    resultado = collection.get(include=["metadatas"])
    fontes: dict[str, dict[str, Any]] = {}
    for metadata in resultado.get("metadatas") or []:
        metadata = metadata or {}
        nome = str(metadata.get("source_file") or "fonte_desconhecida")
        item = fontes.setdefault(
            nome,
            {
                "fonte": nome,
                "caminho": str(metadata.get("source_path") or ""),
                "chunks": 0,
            },
        )
        item["chunks"] += 1
    return sorted(fontes.values(), key=lambda item: item["fonte"].casefold())


def excluir_documento_rag(
    nome_fonte: str,
    mover_para_inativos: bool = True,
) -> dict[str, Any]:
    """Desativa uma fonte e remove todos os seus registros vetoriais."""
    nome_fonte = Path(nome_fonte).name.strip()
    if not nome_fonte:
        raise ValueError("Informe o nome da fonte a excluir.")

    criar_pastas()
    arquivo_ativo = _localizar_arquivo_por_nome(KNOWLEDGE_BASE_DIR, nome_fonte)
    destino: Path | None = None
    if arquivo_ativo is not None:
        if mover_para_inativos:
            destino = _destino_inativo_disponivel(arquivo_ativo.name)
            shutil.move(str(arquivo_ativo), str(destino))
        else:
            arquivo_ativo.unlink()

    collection = obter_collection()
    resultado = collection.get(include=["metadatas"])
    ids_remover = [
        identificador
        for identificador, metadata in zip(
            resultado.get("ids") or [],
            resultado.get("metadatas") or [],
        )
        if _nomes_de_fonte_iguais(
            str((metadata or {}).get("source_file") or ""),
            nome_fonte,
        )
    ]
    if ids_remover:
        collection.delete(ids=ids_remover)

    return {
        "fonte": nome_fonte,
        "chunks_removidos": len(ids_remover),
        "arquivo_encontrado": arquivo_ativo is not None,
        "movido_para": str(destino) if destino else None,
    }


def excluir_documento_indexado(nome_arquivo: str) -> None:
    """Compatibilidade: remove somente os vetores, sem mover o arquivo."""
    collection = obter_collection()
    resultado = collection.get(include=["metadatas"])
    ids_remover = [
        identificador
        for identificador, metadata in zip(
            resultado.get("ids") or [],
            resultado.get("metadatas") or [],
        )
        if _nomes_de_fonte_iguais(
            str((metadata or {}).get("source_file") or ""),
            nome_arquivo,
        )
    ]
    if ids_remover:
        collection.delete(ids=ids_remover)


def _localizar_arquivo_por_nome(pasta: Path, nome_fonte: str) -> Path | None:
    if not pasta.exists():
        return None
    for caminho in pasta.iterdir():
        if caminho.is_file() and _nomes_de_fonte_iguais(caminho.name, nome_fonte):
            return caminho
    return None


def _destino_inativo_disponivel(nome_arquivo: str) -> Path:
    destino = INACTIVE_DOCUMENTS_DIR / nome_arquivo
    if not destino.exists():
        return destino
    indice = 1
    while True:
        candidato = destino.with_name(f"{destino.stem}_{indice}{destino.suffix}")
        if not candidato.exists():
            return candidato
        indice += 1


def _nomes_de_fonte_iguais(nome_a: str, nome_b: str) -> bool:
    return _normalizar_nome_fonte(nome_a) == _normalizar_nome_fonte(nome_b)


def _normalizar_nome_fonte(nome: str) -> str:
    nome = Path(nome).name.strip().casefold()
    nome = unicodedata.normalize("NFKD", nome)
    nome = "".join(caractere for caractere in nome if not unicodedata.combining(caractere))
    return re.sub(r"[^a-z0-9]+", "", nome)


def _gerar_id(nome_arquivo: str, indice: int, chunk: str) -> str:
    digest = hashlib.sha1(f"{nome_arquivo}:{indice}:{chunk}".encode("utf-8")).hexdigest()
    return f"{Path(nome_arquivo).stem}-{indice}-{digest[:12]}"


def _modelo_disponivel(modelo: str, modelos: list[str]) -> bool:
    nomes_aceitos = {modelo}
    if ":" not in modelo:
        nomes_aceitos.add(f"{modelo}:latest")
    return any(item in nomes_aceitos for item in modelos)

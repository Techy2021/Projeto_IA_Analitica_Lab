import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _caminho_configurado(nome_variavel: str, padrao: Path) -> Path:
    """Resolve caminhos relativos do .env a partir da raiz do projeto."""
    valor = os.getenv(nome_variavel)
    if not valor:
        return padrao
    caminho = Path(valor)
    return caminho if caminho.is_absolute() else BASE_DIR / caminho


DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

DB_PATH = _caminho_configurado("DATABASE_PATH", DATA_DIR / "lab_ia.duckdb")

MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"
INACTIVE_DOCUMENTS_DIR = DATA_DIR / "documentos_inativos"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"

MODEL_PATH = _caminho_configurado("MODEL_PATH", MODELS_DIR / "modelo_flaml.pkl")
METADATA_MODEL_PATH = _caminho_configurado(
    "METADATA_PATH",
    MODELS_DIR / "metadata_modelo.json",
)


def criar_pastas():
    """Cria os diretorios necessarios sem sobrescrever artefatos existentes."""
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DATA_DIR.mkdir(exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)
    KNOWLEDGE_BASE_DIR.mkdir(parents=True, exist_ok=True)
    INACTIVE_DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

import duckdb

from app.config import DB_PATH, criar_pastas


def get_connection():
    criar_pastas()
    return duckdb.connect(str(DB_PATH))

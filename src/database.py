import duckdb
import pandas as pd

from src.config import DB_PATH, criar_pastas


def get_connection():
    criar_pastas()
    return duckdb.connect(str(DB_PATH))


def salvar_dataframe(df: pd.DataFrame, nome_tabela: str = "dataset_lab"):
    con = get_connection()

    con.execute(f"DROP TABLE IF EXISTS {nome_tabela}")
    con.register("df_temp", df)
    con.execute(f"CREATE TABLE {nome_tabela} AS SELECT * FROM df_temp")

    con.close()


def carregar_tabela(nome_tabela: str = "dataset_lab") -> pd.DataFrame:
    con = get_connection()

    try:
        df = con.execute(f"SELECT * FROM {nome_tabela}").df()
    finally:
        con.close()

    return df


def consultar_sql(query: str) -> pd.DataFrame:
    con = get_connection()

    try:
        df = con.execute(query).df()
    finally:
        con.close()

    return df


def listar_tabelas():
    con = get_connection()

    try:
        tabelas = con.execute("SHOW TABLES").df()
    finally:
        con.close()

    return tabelas

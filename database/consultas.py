import pandas as pd

from database.conexao import get_connection


def salvar_dataframe(df: pd.DataFrame, nome_tabela: str = "dataset_lab"):
    con = get_connection()
    try:
        con.execute(f"DROP TABLE IF EXISTS {nome_tabela}")
        con.register("df_temp", df)
        con.execute(f"CREATE TABLE {nome_tabela} AS SELECT * FROM df_temp")
    finally:
        con.close()


def carregar_tabela(nome_tabela: str = "dataset_lab") -> pd.DataFrame:
    con = get_connection()
    try:
        return con.execute(f"SELECT * FROM {nome_tabela}").df()
    finally:
        con.close()


def consultar_sql(query: str) -> pd.DataFrame:
    con = get_connection()
    try:
        return con.execute(query).df()
    finally:
        con.close()


def listar_tabelas():
    con = get_connection()
    try:
        return con.execute("SHOW TABLES").df()
    finally:
        con.close()

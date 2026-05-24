import pandas as pd


def carregar_arquivo_upload(arquivo, separador=","):
    nome = arquivo.name.lower()

    if nome.endswith(".csv"):
        return pd.read_csv(arquivo, sep=separador)

    if nome.endswith(".xlsx"):
        return pd.read_excel(arquivo)

    raise ValueError("Formato não suportado. Use CSV ou Excel.")

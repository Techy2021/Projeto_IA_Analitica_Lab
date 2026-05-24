import json
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
from flaml import AutoML
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.config import METADATA_MODEL_PATH, MODEL_PATH, REPORTS_DIR, criar_pastas


RESULTADO_TESTE_PATH = REPORTS_DIR / "resultado_teste_modelo.csv"
METRICAS_MODELO_PATH = REPORTS_DIR / "metricas_modelo.csv"
MATRIZ_CONFUSAO_PATH = REPORTS_DIR / "matriz_confusao.csv"


def _json_default(valor: Any):
    if isinstance(valor, np.integer):
        return int(valor)
    if isinstance(valor, np.floating):
        return float(valor)
    if isinstance(valor, np.ndarray):
        return valor.tolist()
    if isinstance(valor, datetime):
        return valor.isoformat()
    return str(valor)


def _validar_entrada(df: pd.DataFrame, alvo: str, tipo_problema: str, tempo_maximo: int):
    if df.empty:
        raise ValueError("O dataset carregado está vazio.")
    if alvo not in df.columns:
        raise ValueError("A variável alvo selecionada não existe no dataset.")
    if tipo_problema not in {"regression", "classification"}:
        raise ValueError("O tipo de problema deve ser 'regression' ou 'classification'.")
    if tempo_maximo <= 0:
        raise ValueError("O tempo máximo de treinamento deve ser maior que zero.")


def criar_classe_qualidade(df: pd.DataFrame) -> pd.DataFrame:
    if "quality" not in df.columns:
        raise ValueError(
            "A coluna quality não existe no dataset. Ela é necessária para a classificação aprovado/reprovado."
        )

    df_classificacao = df.copy()
    df_classificacao["classe_qualidade"] = np.where(
        df_classificacao["quality"] >= 7,
        "aprovado",
        "reprovado",
    )
    return df_classificacao


def preparar_dados_numericos(
    df: pd.DataFrame,
    alvo: str,
    colunas_remover_entrada: list[str] | None = None,
):
    """Separa alvo e entradas numéricas, imputando ausentes pela mediana."""
    colunas_remover_entrada = colunas_remover_entrada or []
    colunas_preditoras = [
        coluna
        for coluna in df.columns
        if coluna != alvo and coluna not in colunas_remover_entrada
    ]
    X_original = df[colunas_preditoras]
    y = df[alvo]

    linhas_validas = y.notna()
    X_original = X_original.loc[linhas_validas].copy()
    y = y.loc[linhas_validas].copy()

    colunas_numericas = X_original.select_dtypes(include="number").columns.tolist()
    colunas_nao_numericas = [
        coluna for coluna in colunas_preditoras if coluna not in colunas_numericas
    ]

    X = X_original[colunas_numericas].copy()
    colunas_sem_dados = [coluna for coluna in X.columns if X[coluna].isna().all()]
    if colunas_sem_dados:
        X = X.drop(columns=colunas_sem_dados)

    if X.empty:
        raise ValueError(
            "Nenhuma coluna numérica válida foi encontrada para treinar o modelo."
        )

    medianas = X.median(numeric_only=True)
    X = X.fillna(medianas)

    return X, y, colunas_nao_numericas, colunas_sem_dados, medianas


def _calcular_roc_auc(modelo: AutoML, X_test: pd.DataFrame, y_test: pd.Series):
    if not hasattr(modelo, "predict_proba"):
        return None, "O estimador selecionado não disponibiliza predict_proba."

    try:
        probabilidades = modelo.predict_proba(X_test)
        classes = list(getattr(modelo, "classes_", []))

        if len(classes) == 2:
            indice_positivo = classes.index("aprovado") if "aprovado" in classes else 1
            return float(roc_auc_score(y_test, probabilidades[:, indice_positivo])), None

        return (
            float(roc_auc_score(y_test, probabilidades, multi_class="ovr", average="macro")),
            None,
        )
    except Exception as erro:
        return None, f"ROC-AUC não pôde ser calculado: {erro}"


def _salvar_metricas_csv(metricas: dict):
    linhas = [
        {"metrica": metrica, "valor": valor}
        for metrica, valor in metricas.items()
        if isinstance(valor, (int, float)) and valor is not None
    ]
    pd.DataFrame(linhas).to_csv(METRICAS_MODELO_PATH, index=False, encoding="utf-8")


def _metricas_classificacao(y_test: pd.Series, y_pred) -> dict:
    labels = set(y_test.unique().tolist()) | set(pd.Series(y_pred).unique().tolist())
    if {"aprovado", "reprovado"}.issubset(labels):
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(
                precision_score(
                    y_test,
                    y_pred,
                    pos_label="aprovado",
                    zero_division=0,
                )
            ),
            "recall": float(
                recall_score(
                    y_test,
                    y_pred,
                    pos_label="aprovado",
                    zero_division=0,
                )
            ),
            "f1": float(
                f1_score(
                    y_test,
                    y_pred,
                    pos_label="aprovado",
                    zero_division=0,
                )
            ),
        }

    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(
            precision_score(y_test, y_pred, average="macro", zero_division=0)
        ),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
    }


def _serializar_metadata(metadata: dict) -> dict:
    return json.loads(json.dumps(metadata, default=_json_default))


def treinar_modelo_flaml(
    df: pd.DataFrame,
    alvo: str,
    tipo_problema: str | None = None,
    tempo_maximo: int | None = None,
    classificacao_por_qualidade: bool = False,
    test_size: float = 0.2,
    random_state: int = 42,
    tarefa: str | None = None,
    tempo_segundos: int | None = None,
) -> dict:
    """Treina um modelo FLAML AutoML e salva os artefatos de avaliação."""
    criar_pastas()
    tipo_problema = tipo_problema or tarefa
    tempo_maximo = tempo_maximo or tempo_segundos
    linhas_dataset_original = int(df.shape[0])
    colunas_dataset_original = int(df.shape[1])

    colunas_removidas_por_regra = []
    if tipo_problema == "classification" and classificacao_por_qualidade:
        df = criar_classe_qualidade(df)
        alvo = "classe_qualidade"
        colunas_removidas_por_regra = ["quality"]

    _validar_entrada(df, alvo, tipo_problema, tempo_maximo)

    if (
        tipo_problema == "regression"
        and alvo == "indice_qualidade"
        and "classe_qualidade" in df.columns
    ):
        colunas_removidas_por_regra.append("classe_qualidade")

    if (
        tipo_problema == "classification"
        and alvo == "classe_qualidade"
        and "indice_qualidade" in df.columns
    ):
        colunas_removidas_por_regra.append("indice_qualidade")

    colunas_removidas_por_regra = list(dict.fromkeys(colunas_removidas_por_regra))

    X, y, colunas_removidas, colunas_sem_dados, medianas = preparar_dados_numericos(
        df,
        alvo,
        colunas_remover_entrada=colunas_removidas_por_regra,
    )

    if tipo_problema == "classification" and y.nunique() < 2:
        raise ValueError(
            "Há apenas uma classe no conjunto de dados. A classificação precisa de pelo menos duas classes."
        )

    estratificar = None
    if tipo_problema == "classification" and y.value_counts().min() >= 2:
        estratificar = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=estratificar,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
        )

    automl = AutoML()
    automl.fit(
        X_train=X_train,
        y_train=y_train,
        task=tipo_problema,
        time_budget=int(tempo_maximo),
        verbose=0,
    )

    y_pred = automl.predict(X_test)
    matriz_confusao_df = None
    relatorio_classificacao = None
    aviso_roc_auc = None

    if tipo_problema == "regression":
        metricas = {
            "MAE": float(mean_absolute_error(y_test, y_pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "R2": float(r2_score(y_test, y_pred)),
        }
    else:
        roc_auc, aviso_roc_auc = _calcular_roc_auc(automl, X_test, y_test)
        metricas = _metricas_classificacao(y_test, y_pred)
        metricas["roc_auc"] = roc_auc
        relatorio_classificacao = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
        labels = sorted(y.unique().tolist())
        matriz_confusao_df = pd.DataFrame(
            confusion_matrix(y_test, y_pred, labels=labels),
            index=[f"real_{label}" for label in labels],
            columns=[f"previsto_{label}" for label in labels],
        )
        matriz_confusao_df.to_csv(MATRIZ_CONFUSAO_PATH, encoding="utf-8")

    resultado_teste = pd.DataFrame(
        {
            "valor_real": y_test.reset_index(drop=True),
            "valor_previsto": pd.Series(y_pred).reset_index(drop=True),
        }
    )
    resultado_teste.to_csv(RESULTADO_TESTE_PATH, index=False, encoding="utf-8")
    _salvar_metricas_csv(metricas)

    pacote_modelo = {
        "modelo": automl,
        "alvo": alvo,
        "tarefa": tipo_problema,
        "colunas_usadas": X.columns.tolist(),
        "medianas_imputacao": medianas.to_dict(),
    }
    joblib.dump(pacote_modelo, MODEL_PATH)

    metadata = {
        "data_treinamento": datetime.now().isoformat(timespec="seconds"),
        "alvo": alvo,
        "tipo_problema": tipo_problema,
        "classificacao_por_qualidade": bool(classificacao_por_qualidade),
        "regra_classe_qualidade": (
            "quality >= 7 -> aprovado; quality < 7 -> reprovado"
            if classificacao_por_qualidade
            else None
        ),
        "tempo_maximo_segundos": int(tempo_maximo),
        "linhas_dataset_original": linhas_dataset_original,
        "colunas_dataset_original": colunas_dataset_original,
        "linhas_usadas_apos_remover_alvo_nulo": int(len(y)),
        "linhas_treino": int(len(X_train)),
        "linhas_teste": int(len(X_test)),
        "test_size": float(test_size),
        "random_state": int(random_state),
        "melhor_estimador": automl.best_estimator,
        "melhor_configuracao": automl.best_config,
        "melhor_loss_validacao": automl.best_loss,
        "metricas": metricas,
        "aviso_roc_auc": aviso_roc_auc,
        "classification_report": relatorio_classificacao,
        "matriz_confusao": (
            matriz_confusao_df.to_dict() if matriz_confusao_df is not None else None
        ),
        "colunas_usadas": X.columns.tolist(),
        "colunas_removidas_nao_numericas": colunas_removidas,
        "colunas_removidas_por_regra": colunas_removidas_por_regra,
        "colunas_removidas_sem_dados": colunas_sem_dados,
        "medianas_imputacao": medianas.to_dict(),
        "arquivo_modelo": "models/modelo_flaml.pkl",
        "arquivo_metadata": "models/metadata_modelo.json",
        "arquivo_resultado_teste": "reports/resultado_teste_modelo.csv",
        "arquivo_metricas": "reports/metricas_modelo.csv",
        "arquivo_matriz_confusao": (
            "reports/matriz_confusao.csv" if matriz_confusao_df is not None else None
        ),
    }

    metadata_serializavel = _serializar_metadata(metadata)

    with open(METADATA_MODEL_PATH, "w", encoding="utf-8") as arquivo:
        json.dump(metadata_serializavel, arquivo, ensure_ascii=False, indent=2)

    return {
        "automl": automl,
        "metadata": metadata_serializavel,
        "metricas": metricas,
        "resultado_teste": resultado_teste,
        "colunas_usadas": X.columns.tolist(),
        "colunas_removidas_nao_numericas": colunas_removidas,
        "colunas_removidas_por_regra": colunas_removidas_por_regra,
        "colunas_removidas_sem_dados": colunas_sem_dados,
        "modelo_path": MODEL_PATH,
        "metadata_path": METADATA_MODEL_PATH,
        "resultado_teste_path": RESULTADO_TESTE_PATH,
        "metricas_path": METRICAS_MODELO_PATH,
        "matriz_confusao": matriz_confusao_df,
        "matriz_confusao_path": (
            MATRIZ_CONFUSAO_PATH if matriz_confusao_df is not None else None
        ),
    }

import importlib.util
import sys
import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from src.config import criar_pastas
from src.automl_train import treinar_modelo_flaml
from src.database import salvar_dataframe, carregar_tabela, consultar_sql, listar_tabelas
from src.data_loader import carregar_arquivo_upload
from src.predict import (
    carregar_metadata_modelo,
    gerar_previsao,
    modelo_treinado_existe,
    obter_info_modelo,
)


criar_pastas()

st.set_page_config(
    page_title="IA Analítica Laboratorial",
    layout="wide"
)

st.title("IA Analítica Laboratorial")
st.caption("Streamlit + DuckDB + FLAML + Flowise")

menu = st.sidebar.radio(
    "Menu",
    [
        "1. Carregar dados",
        "2. Explorar dados",
        "3. Consultar DuckDB",
        "4. Treinar modelo AutoML",
        "5. Previsão manual",
        "7. Observabilidade",
        "8. Agentes CrewAI + Ollama",
    ]
)


if menu == "1. Carregar dados":
    st.header("Carregar dataset")

    arquivo = st.file_uploader(
        "Carregue um arquivo CSV ou Excel",
        type=["csv", "xlsx"]
    )

    if arquivo is not None:
        separador = ","

        if arquivo.name.lower().endswith(".csv"):
            separador = st.selectbox(
                "Separador do CSV",
                [",", ";", "\t"],
                index=1
            )

        try:
            df = carregar_arquivo_upload(arquivo, separador=separador)

            st.subheader("Prévia dos dados")
            st.dataframe(df.head(), use_container_width=True)

            col1, col2, col3 = st.columns(3)
            col1.metric("Linhas", df.shape[0])
            col2.metric("Colunas", df.shape[1])
            col3.metric("Valores ausentes", int(df.isna().sum().sum()))

            if st.button("Salvar dataset no DuckDB"):
                salvar_dataframe(df, nome_tabela="dataset_lab")
                st.success("Dataset salvo no DuckDB como tabela dataset_lab.")

        except Exception as erro:
            st.error(f"Erro ao carregar arquivo: {erro}")


elif menu == "2. Explorar dados":
    st.header("Explorar dados")

    try:
        df = carregar_tabela("dataset_lab")

        st.subheader("Dimensão")
        st.write(f"{df.shape[0]} linhas x {df.shape[1]} colunas")

        st.subheader("Amostra dos dados")
        st.dataframe(df.head(20), use_container_width=True)

        st.subheader("Tipos de dados")
        tipos = pd.DataFrame({
            "coluna": df.columns,
            "tipo": [str(tipo) for tipo in df.dtypes],
            "nulos": df.isna().sum().values
        })
        st.dataframe(tipos, use_container_width=True)

        st.subheader("Resumo estatístico")
        st.dataframe(df.describe(include="all").T, use_container_width=True)

        colunas_numericas = df.select_dtypes(include="number").columns.tolist()

        if colunas_numericas:
            st.subheader("Distribuição de variável numérica")

            coluna = st.selectbox(
                "Selecione uma coluna",
                colunas_numericas
            )

            fig = px.histogram(
                df,
                x=coluna,
                title=f"Distribuição de {coluna}"
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nenhuma coluna numérica encontrada.")

    except Exception as erro:
        st.warning("Nenhum dataset foi carregado no DuckDB ainda.")
        st.error(erro)


elif menu == "3. Consultar DuckDB":
    st.header("Consulta SQL no DuckDB")

    try:
        st.subheader("Tabelas disponíveis")
        st.dataframe(listar_tabelas(), use_container_width=True)
    except Exception:
        st.info("Nenhuma tabela encontrada ainda.")

    query = st.text_area(
        "Digite uma consulta SQL",
        value="SELECT * FROM dataset_lab LIMIT 10",
        height=150
    )

    if st.button("Executar consulta"):
        try:
            resultado = consultar_sql(query)
            st.dataframe(resultado, use_container_width=True)
        except Exception as erro:
            st.error(f"Erro na consulta SQL: {erro}")


elif menu == "4. Treinar modelo AutoML":
    st.header("Treinar modelo AutoML")

    try:
        df = carregar_tabela("dataset_lab")

        st.subheader("Prévia do dataset carregado")
        st.dataframe(df.head(20), use_container_width=True)

        if df.empty:
            st.warning("A tabela dataset_lab está vazia.")
            st.stop()

        colunas_numericas = df.select_dtypes(include="number").columns.tolist()
        if not colunas_numericas:
            st.warning("Nenhuma coluna numérica foi encontrada no dataset.")
            st.stop()

        col1, col2 = st.columns(2)

        with col1:
            tipo_modelagem = st.radio(
                "Tipo de modelagem",
                [
                    "Regressão",
                    "Classificação",
                ],
            )

        with col2:
            tempo_maximo = st.number_input(
                "Tempo máximo de treinamento (segundos)",
                min_value=5,
                max_value=3600,
                value=30,
                step=5
            )

        tipo_problema = (
            "regression"
            if tipo_modelagem == "Regressão"
            else "classification"
        )
        classificacao_por_qualidade = False

        if tipo_problema == "regression":
            indice_alvo_regressao = (
                colunas_numericas.index("indice_qualidade")
                if "indice_qualidade" in colunas_numericas
                else colunas_numericas.index("quality")
                if "quality" in colunas_numericas
                else 0
            )
            alvo = st.selectbox(
                "Variável alvo da regressão",
                colunas_numericas,
                index=indice_alvo_regressao,
            )
        else:
            colunas_categoricas = df.select_dtypes(
                include=["object", "category", "bool"]
            ).columns.tolist()

            if not colunas_categoricas:
                st.warning(
                    "Nenhuma coluna categórica/textual foi encontrada para classificação."
                )
                st.stop()

            indice_alvo_classificacao = (
                colunas_categoricas.index("classe_qualidade")
                if "classe_qualidade" in colunas_categoricas
                else 0
            )
            alvo = st.selectbox(
                "Variável alvo da classificação",
                colunas_categoricas,
                index=indice_alvo_classificacao,
            )

            if "quality" in df.columns:
                classificacao_por_qualidade = st.checkbox(
                    "Criar classe aprovado/reprovado a partir da coluna quality (Wine Quality)",
                    value=False,
                )
                if classificacao_por_qualidade:
                    alvo = "quality"

        if tipo_problema == "regression":
            st.info(
                "Este modelo tenta prever um índice numérico da amostra a partir dos resultados físico-químicos."
            )
        else:
            st.info(
                "Este modelo aprende uma decisão operacional categórica a partir dos resultados físico-químicos."
            )

            if classificacao_por_qualidade:
                classes_preview = pd.Series(
                    [
                        "aprovado" if valor >= 7 else "reprovado"
                        for valor in df["quality"]
                    ]
                )
                nome_classe_preview = "classe_qualidade"
            else:
                classes_preview = df[alvo].dropna()
                nome_classe_preview = alvo

            if classes_preview.nunique() < 2:
                st.warning(
                    "Há apenas uma classe no conjunto de dados. A classificação precisa de pelo menos duas classes."
                )
                st.stop()

            st.subheader("Distribuição da decisão operacional")
            st.dataframe(
                classes_preview.value_counts()
                .rename_axis(nome_classe_preview)
                .reset_index(name="quantidade"),
                use_container_width=True,
            )

        colunas_removidas_por_regra_preview = []
        if (
            tipo_problema == "regression"
            and alvo == "indice_qualidade"
            and "classe_qualidade" in df.columns
        ):
            colunas_removidas_por_regra_preview.append("classe_qualidade")

        if (
            tipo_problema == "classification"
            and alvo == "classe_qualidade"
            and "indice_qualidade" in df.columns
        ):
            colunas_removidas_por_regra_preview.append("indice_qualidade")

        if classificacao_por_qualidade and "quality" in df.columns:
            colunas_removidas_por_regra_preview.append("quality")

        colunas_removidas_por_regra_preview = list(
            dict.fromkeys(colunas_removidas_por_regra_preview)
        )

        colunas_entrada_numericas = [
            coluna
            for coluna in colunas_numericas
            if coluna != alvo and coluna not in colunas_removidas_por_regra_preview
        ]
        colunas_removidas_preview = [
            coluna
            for coluna in df.columns
            if (
                coluna != alvo
                and coluna not in colunas_entrada_numericas
                and coluna not in colunas_removidas_por_regra_preview
            )
        ]

        st.subheader("Colunas candidatas ao modelo")
        st.write("Colunas numéricas usadas como entrada:")
        st.write(colunas_entrada_numericas)

        if colunas_removidas_preview:
            st.write("Colunas removidas por não serem numéricas:")
            st.write(colunas_removidas_preview)

        if colunas_removidas_por_regra_preview:
            st.write("Colunas removidas por regra de negócio para evitar vazamento:")
            st.write(colunas_removidas_por_regra_preview)

        if st.button("Treinar modelo", type="primary"):
            with st.spinner("Treinando AutoML com FLAML..."):
                resultado = treinar_modelo_flaml(
                    df=df,
                    alvo=alvo,
                    tipo_problema=tipo_problema,
                    tempo_maximo=int(tempo_maximo),
                    classificacao_por_qualidade=classificacao_por_qualidade,
                )

            metadata = resultado["metadata"]
            metricas = resultado["metricas"]
            resultado_teste = resultado["resultado_teste"]

            st.success("Modelo treinado e artefatos salvos com sucesso.")

            st.subheader("Melhor estimador encontrado pelo FLAML")
            st.code(str(metadata["melhor_estimador"]))

            st.subheader("Métricas principais")
            if tipo_problema == "regression":
                m1, m2, m3 = st.columns(3)
                m1.metric("MAE", f"{metricas['MAE']:.4f}")
                m2.metric("RMSE", f"{metricas['RMSE']:.4f}")
                m3.metric("R²", f"{metricas['R2']:.4f}")
            else:
                m1, m2, m3, m4, m5 = st.columns(5)
                m1.metric("Accuracy", f"{metricas['accuracy']:.4f}")
                m2.metric("Precision", f"{metricas['precision']:.4f}")
                m3.metric("Recall", f"{metricas['recall']:.4f}")
                m4.metric("F1", f"{metricas['f1']:.4f}")

                if metricas.get("roc_auc") is not None:
                    m5.metric("ROC-AUC", f"{metricas['roc_auc']:.4f}")
                else:
                    m5.metric("ROC-AUC", "N/D")
                    if metadata.get("aviso_roc_auc"):
                        st.info(metadata["aviso_roc_auc"])

                st.subheader("Matriz de confusão")
                if resultado["matriz_confusao"] is not None:
                    st.dataframe(
                        resultado["matriz_confusao"],
                        use_container_width=True,
                    )

                st.subheader("Classification report")
                st.dataframe(
                    pd.DataFrame(metadata["classification_report"]).T,
                    use_container_width=True
                )

            st.subheader("Valor real x valor previsto")
            st.dataframe(resultado_teste, use_container_width=True)

            if tipo_problema == "regression":
                fig = px.scatter(
                    resultado_teste,
                    x="valor_real",
                    y="valor_previsto",
                    title="Dispersão: valor real x valor previsto",
                    labels={
                        "valor_real": "Valor real",
                        "valor_previsto": "Valor previsto",
                    },
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Colunas usadas no modelo")
            st.write(resultado["colunas_usadas"])

            st.subheader("Colunas removidas")
            st.write("Não numéricas:")
            st.write(resultado["colunas_removidas_nao_numericas"])
            st.write("Por regra de negócio:")
            st.write(resultado["colunas_removidas_por_regra"])
            st.write("Numéricas sem dados válidos:")
            st.write(resultado["colunas_removidas_sem_dados"])

            st.subheader("Metadados completos do treinamento")
            st.json(metadata)

            st.info(
                "Arquivos salvos em models/modelo_flaml.pkl, "
                "models/metadata_modelo.json, reports/resultado_teste_modelo.csv, "
                "reports/metricas_modelo.csv e, para classificação, "
                "reports/matriz_confusao.csv."
            )

    except Exception as erro:
        st.warning("Não foi possível preparar ou treinar o modelo.")
        st.error(f"Detalhes: {erro}")


elif menu == "5. Previsão manual":
    st.header("Previsão manual")

    if not modelo_treinado_existe():
        st.warning(
            "Nenhum modelo treinado encontrado. Treine um modelo antes de gerar previsões."
        )
        st.stop()

    try:
        metadata = carregar_metadata_modelo()
        info_modelo = obter_info_modelo(metadata)
        colunas_usadas = info_modelo["colunas_usadas"]

        col1, col2, col3 = st.columns(3)
        col1.metric("Variável alvo", info_modelo["alvo"])
        col2.metric("Tipo de tarefa", info_modelo["tipo_problema"])
        col3.metric("Melhor estimador", info_modelo["melhor_estimador"])

        valores_medios = {}
        try:
            df = carregar_tabela("dataset_lab")
            valores_medios = (
                df[colunas_usadas]
                .select_dtypes(include="number")
                .mean(numeric_only=True)
                .to_dict()
            )
        except Exception:
            valores_medios = {}

        st.subheader("Informe os valores das variáveis")
        usar_medias = st.checkbox(
            "Usar valores médios do dataset como preenchimento inicial",
            value=True,
            disabled=not bool(valores_medios),
        )

        valores_digitados = {}
        colunas_layout = st.columns(2)

        for indice, coluna in enumerate(colunas_usadas):
            valor_padrao = 0.0
            if usar_medias and coluna in valores_medios and pd.notna(valores_medios[coluna]):
                valor_padrao = float(valores_medios[coluna])

            with colunas_layout[indice % 2]:
                valores_digitados[coluna] = st.number_input(
                    coluna,
                    value=valor_padrao,
                    format="%.6f",
                    key=f"previsao_manual_{coluna}",
                )

        if st.button("Gerar previsão", type="primary"):
            try:
                previsao = gerar_previsao(valores_digitados)

                st.subheader("Resultado previsto")
                if info_modelo["tipo_problema"] == "regression":
                    st.metric("Valor previsto", f"{float(previsao):.3f}")
                else:
                    st.metric("Classe prevista", str(previsao))

            except Exception as erro:
                st.error(f"Erro ao gerar previsão: {erro}")

    except Exception as erro:
        st.error(f"Erro ao carregar informações do modelo: {erro}")


elif menu == "7. Observabilidade":
    st.header("Observabilidade")
    st.write("Monitoramento local das execuções dos agentes CrewAI com Ollama.")

    traces_path = Path("reports") / "traces_crewai_ollama.jsonl"

    if not traces_path.exists():
        st.info("Nenhum trace encontrado ainda em reports/traces_crewai_ollama.jsonl.")
        st.stop()

    registros = []
    linhas_invalidas = 0
    with traces_path.open("r", encoding="utf-8") as arquivo:
        for linha in arquivo:
            linha = linha.strip()
            if not linha:
                continue
            try:
                registros.append(json.loads(linha))
            except json.JSONDecodeError:
                linhas_invalidas += 1

    if not registros:
        st.info("O arquivo de traces existe, mas ainda não possui registros válidos.")
        if linhas_invalidas:
            st.warning(f"Linhas inválidas ignoradas: {linhas_invalidas}")
        st.stop()

    df_traces = pd.DataFrame(registros)

    for coluna in [
        "timestamp",
        "pergunta",
        "resposta",
        "status",
        "agentes",
        "ferramenta_utilizada",
        "modelo_ollama",
        "tempo_execucao_ms",
        "erro",
    ]:
        if coluna not in df_traces.columns:
            df_traces[coluna] = None

    df_traces["tempo_execucao_ms"] = pd.to_numeric(
        df_traces["tempo_execucao_ms"],
        errors="coerce",
    )
    df_traces["agentes_texto"] = df_traces["agentes"].apply(
        lambda valor: ", ".join(valor) if isinstance(valor, list) else str(valor or "")
    )
    df_traces["ferramentas_texto"] = df_traces["ferramenta_utilizada"].apply(
        lambda valor: ", ".join(valor) if isinstance(valor, list) else str(valor or "")
    )

    total_interacoes = len(df_traces)
    total_erros = int((df_traces["status"] == "erro").sum())
    tempo_medio = df_traces["tempo_execucao_ms"].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de interações", total_interacoes)
    col2.metric("Erros", total_erros)
    col3.metric(
        "Tempo médio",
        "N/D" if pd.isna(tempo_medio) else f"{tempo_medio / 1000:.1f} s",
    )

    agentes_series = (
        df_traces["agentes"]
        .explode()
        .dropna()
        .astype(str)
    )
    st.subheader("Agentes mais usados")
    if agentes_series.empty:
        st.info("Nenhum agente registrado nos traces.")
    else:
        st.dataframe(
            agentes_series.value_counts()
            .rename_axis("agente")
            .reset_index(name="quantidade"),
            use_container_width=True,
        )

    st.subheader("Traces")
    colunas_tabela = [
        "timestamp",
        "status",
        "modelo_ollama",
        "tempo_execucao_ms",
        "agentes_texto",
        "ferramentas_texto",
        "pergunta",
        "resposta",
        "erro",
    ]
    st.dataframe(
        df_traces[colunas_tabela].sort_index(ascending=False),
        use_container_width=True,
    )

    csv_traces = df_traces[colunas_tabela].to_csv(index=False).encode("utf-8-sig")
    st.download_button(
        "Baixar traces em CSV",
        data=csv_traces,
        file_name="traces_crewai_ollama.csv",
        mime="text/csv",
    )

    if linhas_invalidas:
        st.warning(f"Linhas inválidas ignoradas: {linhas_invalidas}")


elif menu == "8. Agentes CrewAI + Ollama":
    st.header("Agentes CrewAI com Ollama")
    st.write("Esta aba executa agentes CrewAI usando Ollama local como LLM.")
    st.warning(
        "Mantenha a API FastAPI na porta 8000 e o Ollama ativo antes de usar os agentes."
    )

    diagnostico = {
        "Python executável": sys.executable,
        "Python versão": sys.version,
        "CrewAI encontrado": "Sim" if importlib.util.find_spec("crewai") else "Não",
        "LiteLLM encontrado": "Sim" if importlib.util.find_spec("litellm") else "Não",
        "Requests encontrado": "Sim" if importlib.util.find_spec("requests") else "Não",
        "Ollama ativo": "Não",
        "Modelo Ollama configurado": "Indefinido",
        "API FastAPI ativa": "Não",
    }
    comando_instalacao_crewai = (
        "D:\n"
        "cd D:\\Projeto_IA_Analitica_Lab\n"
        ".\\.venv\\Scripts\\Activate.ps1\n"
        "python -m pip install crewai crewai-tools litellm requests python-dotenv"
    )

    try:
        from src.ollama_check import (
            modelo_ollama_disponivel,
            obter_config_ollama,
            verificar_ollama,
        )

        ollama_base_url, ollama_model = obter_config_ollama()
        diagnostico["Modelo Ollama configurado"] = ollama_model
        status_ollama = verificar_ollama()
        if status_ollama.get("status") == "ok":
            diagnostico["Ollama ativo"] = "Sim"
            st.success(f"Ollama ativo em {ollama_base_url}.")
        else:
            st.error(status_ollama.get("mensagem"))

        status_modelo = modelo_ollama_disponivel(ollama_model)
        if status_modelo.get("status") == "ok":
            st.success(f"Modelo Ollama configurado: {ollama_model}.")
        else:
            st.error(status_modelo.get("mensagem"))
    except Exception as erro:
        ollama_model = "llama3.2:3b"
        diagnostico["Modelo Ollama configurado"] = ollama_model
        st.error(f"Não foi possível verificar o Ollama: {erro}")

    try:
        from src.crewai_tools_lab import obter_api_base_url

        api_base_url = obter_api_base_url()
        resposta_health = requests.get(f"{api_base_url}/health", timeout=3)
        if resposta_health.ok:
            diagnostico["API FastAPI ativa"] = "Sim"
            st.success(f"API FastAPI local ativa em {api_base_url}.")
        else:
            st.warning(f"API FastAPI respondeu com status {resposta_health.status_code}.")
    except requests.exceptions.RequestException:
        st.error(
            "API local não encontrada. Inicie com: "
            "python -m uvicorn api.model_api:app --reload --port 8000"
        )

    with st.expander("Diagnóstico do ambiente", expanded=False):
        st.write("Python executável:", diagnostico["Python executável"])
        st.write("Python versão:", diagnostico["Python versão"])
        st.write("CrewAI encontrado:", diagnostico["CrewAI encontrado"])
        st.write("LiteLLM encontrado:", diagnostico["LiteLLM encontrado"])
        st.write("Requests encontrado:", diagnostico["Requests encontrado"])
        st.write("Ollama ativo:", diagnostico["Ollama ativo"])
        st.write("Modelo Ollama configurado:", diagnostico["Modelo Ollama configurado"])
        st.write("API FastAPI ativa:", diagnostico["API FastAPI ativa"])
        if diagnostico["CrewAI encontrado"] == "Não":
            st.error("CrewAI não encontrado no ambiente virtual atual.")
            st.code(comando_instalacao_crewai, language="powershell")

    st.subheader("Exemplos de perguntas")
    exemplos = [
        (
            "Faça uma previsão para umidade 12.5, proteína 46.2, extrato etéreo 1.8, "
            "fibras 4.7, matéria mineral 4.5, urease 0.12 e solubilidade 82."
        ),
        "Quantas amostras existem por classe de qualidade?",
        "Quais são as médias por classe?",
        "Explique as métricas do modelo atual.",
        "Quais colunas foram usadas no treinamento?",
    ]
    for exemplo in exemplos:
        st.markdown(f"- {exemplo}")

    pergunta = st.text_area(
        "Digite sua pergunta para os agentes CrewAI",
        height=140,
        placeholder=exemplos[0],
    )

    if st.button("Executar agentes", type="primary"):
        if not pergunta.strip():
            st.warning("Digite uma pergunta antes de executar os agentes.")
        else:
            try:
                from src.crewai_agents_lab import executar_crew_lab

                with st.spinner("Executando agentes CrewAI com Ollama..."):
                    resultado = executar_crew_lab(pergunta)

                st.subheader("Resposta final")
                if resultado.get("status") == "ok":
                    st.success("Execução concluída.")
                else:
                    st.error("Execução não concluída.")

                st.write(resultado.get("resposta", "Sem resposta."))
                st.write("Status:", resultado.get("status", "indefinido"))
                st.write("Modelo Ollama:", resultado.get("modelo_ollama", ollama_model))
                if resultado.get("status") != "ok" and resultado.get("traceback"):
                    with st.expander("Erro técnico"):
                        st.code(resultado["traceback"])

                agentes = resultado.get("agentes", [])
                if agentes:
                    st.write("Agentes usados:")
                    st.write(agentes)
                else:
                    st.info(
                        "Nenhum agente foi executado. Verifique se CrewAI está instalado, "
                        "se Ollama está ativo e se o modelo local foi baixado."
                    )
            except Exception as erro:
                st.error(
                    "Não foi possível carregar a camada CrewAI. Veja o erro real abaixo."
                )
                st.exception(erro)

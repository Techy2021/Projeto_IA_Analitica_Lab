import importlib.util
import sys
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px
import requests

from app.config import BASE_DIR, KNOWLEDGE_BASE_DIR, REPORTS_DIR, criar_pastas
from app.components.navigation import navigate
from ai.modeling.automl_train import treinar_modelo_flaml
from ai.modeling.predict import (
    carregar_metadata_modelo,
    gerar_previsao,
    modelo_treinado_existe,
    obter_info_modelo,
)
from ai.rag.index import (
    contar_documentos_indexados,
    excluir_documento_rag,
    indexar_documentos,
    verificar_modelo_embedding,
)
from ai.rag.loader import EXTENSOES_SUPORTADAS, carregar_documentos, salvar_upload
from ai.rag.retriever import buscar_trechos
from ai.rag.router import (
    classificar_pergunta,
    formatar_resposta_numerica,
    responder_pergunta_numerica,
)
from data.loader import carregar_arquivo_upload
from database.consultas import (
    carregar_tabela,
    consultar_sql,
    listar_tabelas,
    salvar_dataframe,
)


criar_pastas()

st.set_page_config(
    page_title="Laboratory AI Intelligence Platform",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    :root { --bg:#F8FAFC; --sidebar:#FFFFFF; --card:#FFFFFF; --cyan:#0284C7; --violet:#7C3AED; --text:#0F172A; --muted:#475569; --border:rgba(15,23,42,.12); --soft:#E0F2FE; --font-scale:1.14; }
    html, body, [class*="css"] { font-family:"Inter",sans-serif; font-size:calc(16px * var(--font-scale)); }
    .stApp, [data-testid="stAppViewContainer"] { background:var(--bg); color:var(--text); }
    [data-testid="stHeader"] { background:rgba(248,250,252,.86); backdrop-filter:blur(18px); }
    [data-testid="stSidebar"] { background:var(--sidebar); border-right:1px solid var(--border); box-shadow:8px 0 30px rgba(15,23,42,.05); }
    [data-testid="stSidebar"] > div:first-child { padding-top:1rem; }
    [data-testid="stSidebar"] .stRadio label { padding:.6rem .72rem; border-radius:8px; transition:.2s ease; }
    [data-testid="stSidebar"] .stRadio label:hover { background:rgba(2,132,199,.08); color:var(--cyan); }
    [data-testid="stSidebar"] .stRadio label:has(input:checked) { color:var(--cyan); background:rgba(2,132,199,.12); border:1px solid rgba(2,132,199,.18); }
    .block-container { max-width:1500px; padding:5.25rem 2rem 3rem; }
    .platform-topbar { position:fixed; top:0; left:0; right:0; height:64px; z-index:999; display:flex; align-items:center; justify-content:space-between; padding:0 2rem 0 calc(21rem + 1rem); background:rgba(255,255,255,.9); backdrop-filter:blur(18px); border-bottom:1px solid var(--border); box-shadow:0 8px 26px rgba(15,23,42,.05); }
    .topbar-brand { color:var(--text); font-size:calc(.98rem * var(--font-scale)); font-weight:700; } .topbar-brand span { color:var(--cyan); }
    .topbar-actions { display:flex; align-items:center; gap:.7rem; color:var(--muted); font-size:calc(.82rem * var(--font-scale)); }
    .status-dot { width:7px; height:7px; background:#22C55E; border-radius:50%; box-shadow:0 0 10px #22C55E; }
    .sidebar-brand { padding:.6rem .45rem 1.25rem; }
    .sidebar-logo { width:38px; height:38px; display:inline-flex; align-items:center; justify-content:center; border-radius:9px; color:#FFFFFF; font-weight:900; background:linear-gradient(135deg,var(--cyan),#38BDF8); box-shadow:0 0 24px rgba(2,132,199,.18); }
    .sidebar-name { display:inline-block; vertical-align:middle; margin-left:.65rem; color:var(--text); font-weight:700; } .sidebar-name small { display:block; color:var(--muted); font-size:calc(.68rem * var(--font-scale)); font-weight:600; margin-top:.1rem; }
    h1,h2,h3 { color:var(--text)!important; letter-spacing:0; } h1 { font-size:calc(1.9rem * var(--font-scale))!important; font-weight:750!important; } h2 { font-size:calc(1.34rem * var(--font-scale))!important; margin-top:1.25rem!important; }
    p,label,.stCaption { color:var(--muted); }
    .page-kicker { color:var(--cyan); font-size:calc(.78rem * var(--font-scale)); font-weight:800; text-transform:uppercase; letter-spacing:.12em; }
    .page-title { font-size:calc(1.86rem * var(--font-scale)); font-weight:750; color:var(--text); margin:.25rem 0 .3rem; }
    .page-description { color:var(--muted); font-size:calc(1rem * var(--font-scale)); margin-bottom:1.6rem; }
    .hero { padding:2rem 2.1rem; border-radius:8px; margin-bottom:1.4rem; background:radial-gradient(circle at 80% 10%,rgba(124,58,237,.13),transparent 32%),radial-gradient(circle at 55% 120%,rgba(2,132,199,.16),transparent 38%),linear-gradient(135deg,#FFFFFF,#EFF6FF); border:1px solid rgba(2,132,199,.18); box-shadow:0 18px 45px rgba(15,23,42,.10); animation:fadeUp .45s ease both; }
    .hero-badge { color:var(--cyan); font-size:calc(.78rem * var(--font-scale)); font-weight:800; text-transform:uppercase; letter-spacing:.13em; } .hero h1 { font-size:calc(2.25rem * var(--font-scale))!important; max-width:760px; margin:.7rem 0 .6rem; } .hero p { max-width:760px; font-size:calc(1.04rem * var(--font-scale)); line-height:1.65; color:var(--muted); }
    .metric-card,[data-testid="stMetric"] { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1rem 1.05rem; box-shadow:0 8px 30px rgba(15,23,42,.08); transition:.2s ease; animation:fadeUp .45s ease both; }
    .metric-card:hover,[data-testid="stMetric"]:hover { transform:translateY(-2px); border-color:rgba(2,132,199,.35); box-shadow:0 10px 32px rgba(2,132,199,.12); }
    .metric-label { color:var(--muted); font-size:calc(.76rem * var(--font-scale)); font-weight:700; text-transform:uppercase; letter-spacing:.04em; } .metric-value { color:var(--text); font-size:calc(1.75rem * var(--font-scale)); font-weight:750; margin:.35rem 0 .15rem; } .metric-foot { color:#64748B; font-size:calc(.72rem * var(--font-scale)); }
    [data-testid="stMetricLabel"] { color:var(--muted); font-size:calc(.92rem * var(--font-scale)); } [data-testid="stMetricValue"] { color:var(--text); font-size:calc(1.72rem * var(--font-scale)); }
    [data-testid="stVerticalBlockBorderWrapper"] { background:#FFFFFF; border-color:var(--border)!important; border-radius:8px; }
    .model-card { background:var(--card); border:1px solid var(--border); border-radius:8px; padding:1.15rem; transition:.2s ease; min-height:150px; box-shadow:0 8px 30px rgba(15,23,42,.07); } .model-card:hover { border-color:rgba(2,132,199,.35); box-shadow:0 8px 28px rgba(2,132,199,.10); }
    .status-pill { display:inline-block; padding:.24rem .48rem; border-radius:99px; font-size:calc(.7rem * var(--font-scale)); font-weight:800; background:rgba(22,163,74,.10); color:#15803D; border:1px solid rgba(22,163,74,.18); }
    .model-name { color:var(--text); font-weight:700; font-size:calc(1.12rem * var(--font-scale)); margin:.8rem 0 .35rem; } .model-meta { color:var(--muted); font-size:calc(.82rem * var(--font-scale)); line-height:1.7; }
    .stButton>button,.stDownloadButton>button { border-radius:7px; border:1px solid rgba(2,132,199,.28); background:rgba(2,132,199,.08); color:var(--cyan); font-weight:700; transition:.2s ease; font-size:calc(.94rem * var(--font-scale)); }
    .stButton>button:hover,.stDownloadButton>button:hover { border-color:var(--cyan); background:rgba(2,132,199,.14); box-shadow:0 0 20px rgba(2,132,199,.14); }
    .stButton>button[kind="primary"] { background:var(--cyan); color:#FFFFFF; border-color:var(--cyan); }
    input,textarea,[data-baseweb="select"]>div { background:#FFFFFF!important; color:var(--text)!important; border-color:var(--border)!important; font-size:calc(.96rem * var(--font-scale)); }
    [data-testid="stFileUploaderDropzone"] { background:#FFFFFF; border-color:rgba(2,132,199,.25); border-radius:8px; }
    [data-testid="stDataFrame"],[data-testid="stPlotlyChart"] { border:1px solid var(--border); border-radius:8px; overflow:hidden; }
    [data-testid="stExpander"] { background:#FFFFFF; border-color:var(--border); border-radius:8px; }
    [data-testid="stChatMessage"] { background:#FFFFFF; border:1px solid var(--border); border-radius:8px; padding:.85rem 1rem; box-shadow:0 6px 24px rgba(15,23,42,.06); }
    [data-testid="stChatInput"] { background:#FFFFFF; border-color:rgba(2,132,199,.25); }
    .section-label { color:var(--text); font-size:calc(1.04rem * var(--font-scale)); font-weight:800; margin:1.4rem 0 .75rem; } .empty-state { padding:2rem; text-align:center; color:var(--muted); background:#FFFFFF; border:1px dashed rgba(15,23,42,.18); border-radius:8px; }
    @keyframes fadeUp { from{opacity:0;transform:translateY(6px)} to{opacity:1;transform:translateY(0)} }
    @media(max-width:900px){.platform-topbar{padding-left:4.5rem}.block-container{padding:4.8rem 1rem 2rem}.hero{padding:1.35rem}.hero h1{font-size:1.55rem!important}}
    </style>
    """,
    unsafe_allow_html=True,
)


def page_header(kicker, title, description):
    st.markdown(f'<div class="page-kicker">{kicker}</div><div class="page-title">{title}</div><div class="page-description">{description}</div>', unsafe_allow_html=True)


def html_metric(label, value, foot):
    return f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-foot">{foot}</div></div>'


def load_platform_stats():
    stats = {"datasets":0, "rows":0, "models":0, "predictions":0, "accuracy":"N/D", "docs":0, "vectors":0, "agents":0}
    try:
        tables = listar_tabelas()
        stats["datasets"] = len(tables)
        if "dataset_lab" in tables.astype(str).values:
            stats["rows"] = len(carregar_tabela("dataset_lab"))
    except Exception:
        pass
    try:
        metadata = carregar_metadata_modelo()
        stats["models"] = 1
        score = metadata.get("metricas", {}).get("accuracy", metadata.get("metricas", {}).get("R2"))
        stats["accuracy"] = f"{score * 100:.1f}%" if score is not None else "N/D"
    except Exception:
        pass
    try:
        stats["docs"] = len(carregar_documentos())
        stats["vectors"] = contar_documentos_indexados()
    except Exception:
        pass
    traces_path = REPORTS_DIR / "traces_crewai_ollama.jsonl"
    if traces_path.exists():
        try:
            traces = [json.loads(line) for line in traces_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            stats["agents"] = len({agent for trace in traces for agent in trace.get("agentes", [])})
            stats["predictions"] = sum("previs" in str(trace.get("pergunta", "")).lower() for trace in traces)
        except Exception:
            pass
    return stats


PREDICTIONS_HISTORY_PATH = REPORTS_DIR / "predictions_history.jsonl"
CHAT_HISTORY_PATH = REPORTS_DIR / "assistant_chat_history.jsonl"


def append_prediction_history(payload):
    PREDICTIONS_HISTORY_PATH.parent.mkdir(exist_ok=True)
    with PREDICTIONS_HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")


def load_prediction_history():
    if not PREDICTIONS_HISTORY_PATH.exists():
        return pd.DataFrame()
    records = []
    with PREDICTIONS_HISTORY_PATH.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(records)


def append_chat_history(role, content):
    CHAT_HISTORY_PATH.parent.mkdir(exist_ok=True)
    with CHAT_HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps(
                {"timestamp": datetime.now().isoformat(timespec="seconds"), "role": role, "content": content},
                ensure_ascii=False,
            )
            + "\n"
        )


def delete_knowledge_document(documento):
    base_dir = KNOWLEDGE_BASE_DIR.resolve()
    file_path = Path(documento.caminho).resolve()
    file_path.relative_to(base_dir)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Arquivo não encontrado: {documento.nome_arquivo}")
    resultado = excluir_documento_rag(
        documento.nome_arquivo,
        mover_para_inativos=True,
    )
    return (
        "Documento movido para data/documentos_inativos e "
        f"{resultado['chunks_removidos']} vetor(es) removido(s)."
    )


def runtime_status():
    status = {
        "FastAPI": ("Offline", "http://localhost:8000"),
        "Ollama": ("Offline", "http://localhost:11434"),
        "Modelo preditivo": ("Pronto" if modelo_treinado_existe() else "Ausente", "models/modelo_flaml.pkl"),
        "DuckDB": ("Offline", "data/lab_ia.duckdb"),
        "RAG": ("0 vetores", "vectorstore"),
    }
    try:
        from ai.agentes.crewai_tools_lab import obter_api_base_url

        api_url = obter_api_base_url()
        if requests.get(f"{api_url}/health", timeout=2).ok:
            status["FastAPI"] = ("Online", api_url)
    except Exception:
        pass
    try:
        from ai.agentes.ollama_check import obter_config_ollama

        ollama_url, ollama_model = obter_config_ollama()
        if requests.get(f"{ollama_url}/api/tags", timeout=2).ok:
            status["Ollama"] = ("Online", ollama_model)
    except Exception:
        pass
    try:
        listar_tabelas()
        status["DuckDB"] = ("Online", "data/lab_ia.duckdb")
    except Exception:
        pass
    try:
        status["RAG"] = (f"{contar_documentos_indexados()} vetores", "ChromaDB local")
    except Exception:
        pass
    return status


st.markdown('<div class="platform-topbar"><div class="topbar-brand">Laboratory AI <span>/ Intelligence Platform</span></div><div class="topbar-actions"><span class="status-dot"></span><span>Platform operational</span><span>•</span><span>Industrial Lab</span></div></div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-brand"><span class="sidebar-logo">AI</span><span class="sidebar-name">Lab Intelligence<small>ENTERPRISE PLATFORM</small></span></div>', unsafe_allow_html=True)
nav_options = ["📊 Dashboard","📂 Dados","🤖 Modelos IA","📈 Predições","🧠 Assistente IA","📚 Base de Conhecimento","🔎 RAG","📜 Observabilidade","⚙️ Configurações"]
data_options = ["Carregar dataset","Explorar dados","SQL Studio"]
st.session_state.setdefault("active_area", nav_options[0])
st.session_state.setdefault("active_data_module", data_options[0])
st.sidebar.caption("WORKSPACE")
for index, option in enumerate(nav_options):
    is_active = st.session_state["active_area"] == option
    if st.sidebar.button(
        option,
        key=f"nav_button_{index}",
        type="primary" if is_active else "secondary",
        use_container_width=True,
    ):
        st.session_state["active_area"] = option
        st.rerun()

area = st.session_state["active_area"]
submenu = None
if area == "📂 Dados":
    st.sidebar.caption("MÓDULO DE DADOS")
    for index, option in enumerate(data_options):
        is_active = st.session_state["active_data_module"] == option
        if st.sidebar.button(
            f"↳ {option}",
            key=f"data_button_{index}",
            type="primary" if is_active else "secondary",
            use_container_width=True,
        ):
            st.session_state["active_data_module"] = option
            st.rerun()
    submenu = st.session_state["active_data_module"]
menu_map = {"📊 Dashboard":"0. Dashboard","🤖 Modelos IA":"4. Treinar modelo AutoML","📈 Predições":"5. Previsão manual","🧠 Assistente IA":"8. Agentes CrewAI + Ollama","📚 Base de Conhecimento":"9. Base de Conhecimento / RAG","🔎 RAG":"9. Base de Conhecimento / RAG","📜 Observabilidade":"7. Observabilidade","⚙️ Configurações":"10. Configurações"}
data_map = {"Carregar dataset":"1. Carregar dados","Explorar dados":"2. Explorar dados","SQL Studio":"3. Consultar DuckDB"}
menu = data_map.get(submenu, menu_map.get(area))
st.sidebar.markdown("---")
st.sidebar.caption("APARÊNCIA")
theme_mode = st.sidebar.selectbox(
    "Tema",
    ["Claro", "Escuro", "Sistema"],
    index=["Claro", "Escuro", "Sistema"].index(st.session_state.get("theme_mode", "Claro")),
    key="theme_mode",
)
font_scale = st.sidebar.slider(
    "Tamanho da fonte",
    min_value=1.0,
    max_value=1.4,
    value=float(st.session_state.get("font_scale", 1.14)),
    step=0.02,
    key="font_scale",
)
st.markdown(f"<style>:root {{ --font-scale:{font_scale}; }}</style>", unsafe_allow_html=True)
if theme_mode in {"Escuro", "Sistema"}:
    st.markdown(
        """
        <style>
        :root { --bg:#0B1020; --sidebar:#111827; --card:#1E293B; --cyan:#00D4FF; --text:#F8FAFC; --muted:#CBD5E1; --border:rgba(148,163,184,.22); }
        [data-testid="stHeader"], .platform-topbar { background:rgba(11,16,32,.9)!important; box-shadow:none!important; }
        [data-testid="stSidebar"] { background:#111827!important; box-shadow:none!important; }
        .hero { background:radial-gradient(circle at 80% 10%,rgba(124,58,237,.32),transparent 32%),radial-gradient(circle at 55% 120%,rgba(0,212,255,.16),transparent 38%),#111A2E!important; box-shadow:0 18px 55px rgba(0,0,0,.22)!important; }
        input, textarea, [data-baseweb="select"]>div, [data-testid="stFileUploaderDropzone"], [data-testid="stChatMessage"], [data-testid="stExpander"], .empty-state { background:#111827!important; color:#F8FAFC!important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
st.sidebar.markdown("---")
st.sidebar.caption("LOCAL AI RUNTIME")
runtime = runtime_status()
fastapi_label = runtime["FastAPI"][0]
st.sidebar.markdown(f'<span class="status-pill">● {fastapi_label.upper()}</span>', unsafe_allow_html=True)
with st.sidebar.expander("Status central", expanded=False):
    for service, (state, detail) in runtime.items():
        icon = "✅" if state in {"Online", "Pronto"} or "vetores" in state else "⚠️"
        st.write(f"{icon} **{service}**: {state}")
        st.caption(detail)


if menu == "0. Dashboard":
    stats = load_platform_stats()
    st.markdown(
        '<div class="hero"><div class="hero-badge">Enterprise AI workspace</div>'
        '<h1>Laboratory AI Intelligence Platform</h1>'
        '<p>Centralize dados laboratoriais, treine modelos de IA, execute predições e converse com agentes inteligentes.</p></div>',
        unsafe_allow_html=True,
    )
    action1, action2, _ = st.columns([1.1, 1.1, 4])
    if action1.button("＋ Carregar Dataset", type="primary", use_container_width=True):
        navigate("📂 Dados", "Carregar dataset")
    if action2.button("✦ Conversar com IA", use_container_width=True):
        navigate("🧠 Assistente IA")

    cols = st.columns(6)
    metrics = [
        ("Datasets carregados", stats["datasets"], f'{stats["rows"]:,} registros disponíveis'),
        ("Modelos treinados", stats["models"], "Artefatos prontos para uso"),
        ("Predições realizadas", stats["predictions"], "Interações rastreadas"),
        ("Precisão média", stats["accuracy"], "Melhor score do modelo"),
        ("Documentos RAG", stats["docs"], f'{stats["vectors"]} vetores indexados'),
        ("Agentes IA", stats["agents"], "Agentes observados"),
    ]
    for col, metric in zip(cols, metrics):
        col.markdown(html_metric(*metric), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Status central da plataforma</div>', unsafe_allow_html=True)
    status_cols = st.columns(5)
    for col, (service, (state, detail)) in zip(status_cols, runtime_status().items()):
        col.markdown(html_metric(service, state, detail), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Visão operacional</div>', unsafe_allow_html=True)
    left, right = st.columns([1.65, 1])
    with left:
        try:
            df_dash = carregar_tabela("dataset_lab")
            numeric = df_dash.select_dtypes(include="number").columns.tolist()
            chart_df = df_dash[numeric[:4]].mean().reset_index()
            chart_df.columns = ["Indicador", "Média"]
            fig = px.bar(chart_df, x="Indicador", y="Média", color="Indicador", color_discrete_sequence=["#00D4FF", "#7C3AED", "#22C55E", "#F59E0B"], title="Perfil médio dos indicadores laboratoriais")
            fig.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#0F172A", showlegend=False, margin=dict(l=20, r=20, t=55, b=20))
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.markdown('<div class="empty-state">Carregue um dataset para visualizar indicadores operacionais.</div>', unsafe_allow_html=True)
    with right:
        st.markdown(
            '<div class="model-card"><span class="status-pill">● PRODUCTION</span>'
            '<div class="model-name">AutoML Quality Predictor</div>'
            f'<div class="model-meta">Engine: FLAML<br>Score atual: {stats["accuracy"]}<br>Vetores RAG: {stats["vectors"]}<br>Atualização: {datetime.now().strftime("%d/%m/%Y %H:%M")}</div></div>',
            unsafe_allow_html=True,
        )
        st.markdown("##### Ações rápidas")
        if st.button("Executar nova predição", use_container_width=True):
            navigate("📈 Predições")
        if st.button("Abrir SQL Studio", use_container_width=True):
            navigate("📂 Dados", "SQL Studio")
        if st.button("Reindexar documentos", use_container_width=True):
            navigate("🔎 RAG")


elif menu == "1. Carregar dados":
    page_header("Data workspace", "Carregar dataset", "Importe dados laboratoriais em CSV ou Excel e publique-os no DuckDB.")

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

            st.subheader("Validação antes da importação")
            val1, val2, val3 = st.columns(3)
            val1.metric("Duplicados", int(df.duplicated().sum()))
            val2.metric("Colunas numéricas", len(df.select_dtypes(include="number").columns))
            val3.metric("Colunas texto/categoria", len(df.select_dtypes(exclude="number").columns))
            validation_table = pd.DataFrame(
                {
                    "coluna": df.columns,
                    "tipo": [str(dtype) for dtype in df.dtypes],
                    "nulos": df.isna().sum().values,
                    "nulos_pct": (df.isna().mean().values * 100).round(2),
                }
            )
            with st.expander("Ver diagnóstico de colunas", expanded=False):
                st.dataframe(validation_table, use_container_width=True)

            confirmar_importacao = st.checkbox("Confirmo a importação deste dataset para a tabela dataset_lab.")
            if st.button("Salvar dataset no DuckDB", disabled=not confirmar_importacao):
                salvar_dataframe(df, nome_tabela="dataset_lab")
                st.success("Dataset salvo no DuckDB como tabela dataset_lab.")

        except Exception as erro:
            st.error(f"Erro ao carregar arquivo: {erro}")


elif menu == "2. Explorar dados":
    page_header("Data workspace", "Explorar dados", "Analise estrutura, qualidade e distribuição das variáveis laboratoriais.")

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
    page_header("Data workspace", "SQL Studio", "Consulte os dados laboratoriais diretamente no motor analítico DuckDB.")

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
    page_header("AI model registry", "Modelos IA", "Treine, avalie e publique modelos preditivos com AutoML.")

    if modelo_treinado_existe():
        try:
            current_model = carregar_metadata_modelo()
            score = current_model.get("metricas", {}).get("accuracy", current_model.get("metricas", {}).get("R2"))
            score_text = f"{score:.4f}" if score is not None else "N/D"
            m1, m2 = st.columns(2)
            m1.markdown(
                '<div class="model-card"><span class="status-pill">● PRODUCTION</span>'
                f'<div class="model-name">{current_model.get("melhor_estimador", "AutoML model").upper()}</div>'
                f'<div class="model-meta">Alvo: {current_model.get("alvo", "N/D")}<br>Score: {score_text}<br>Última execução: {str(current_model.get("data_treinamento", "N/D")).replace("T", " ")}</div></div>',
                unsafe_allow_html=True,
            )
            m2.markdown(
                '<div class="model-card"><span class="status-pill">● READY</span>'
                '<div class="model-name">Novo experimento AutoML</div>'
                '<div class="model-meta">Configure o problema, selecione o alvo e execute uma nova rodada de treinamento controlado.</div></div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="section-label">Configurar novo treinamento</div>', unsafe_allow_html=True)
        except Exception:
            pass

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
    page_header("Inference workspace", "Predições", "Execute inferências manuais com o modelo atualmente publicado.")

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
                prediction_record = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "alvo": info_modelo["alvo"],
                    "tipo_problema": info_modelo["tipo_problema"],
                    "estimador": info_modelo["melhor_estimador"],
                    "entradas": valores_digitados,
                    "previsao": float(previsao) if info_modelo["tipo_problema"] == "regression" else str(previsao),
                }
                append_prediction_history(prediction_record)

                st.subheader("Resultado previsto")
                if info_modelo["tipo_problema"] == "regression":
                    st.metric("Valor previsto", f"{float(previsao):.3f}")
                else:
                    st.metric("Classe prevista", str(previsao))
                st.success("Predição registrada no histórico.")

            except Exception as erro:
                st.error(f"Erro ao gerar previsão: {erro}")

        prediction_history = load_prediction_history()
        if not prediction_history.empty:
            st.subheader("Histórico de predições")
            history_view = prediction_history.copy()
            if "entradas" in history_view.columns:
                history_view["entradas"] = history_view["entradas"].apply(
                    lambda value: json.dumps(value, ensure_ascii=False) if isinstance(value, dict) else value
                )
            st.dataframe(history_view.sort_values("timestamp", ascending=False).head(50), use_container_width=True)
            st.download_button(
                "Baixar histórico em CSV",
                data=history_view.to_csv(index=False).encode("utf-8-sig"),
                file_name="historico_predicoes.csv",
                mime="text/csv",
            )

    except Exception as erro:
        st.error(f"Erro ao carregar informações do modelo: {erro}")


elif menu == "7. Observabilidade":
    page_header("Tracing & analytics", "Observabilidade", "Monitore interações, latência, erros e uso dos agentes de IA.")

    traces_path = REPORTS_DIR / "traces_crewai_ollama.jsonl"

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

    st.markdown('<div class="section-label">Filtros</div>', unsafe_allow_html=True)
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    status_options = ["Todos"] + sorted(df_traces["status"].dropna().astype(str).unique().tolist())
    with filter_col1:
        status_filter = st.selectbox("Status", status_options)
    agent_values = sorted(df_traces["agentes"].explode().dropna().astype(str).unique().tolist())
    with filter_col2:
        agent_filter = st.selectbox("Agente", ["Todos"] + agent_values)
    with filter_col3:
        model_filter = st.selectbox("Modelo", ["Todos"] + sorted(df_traces["modelo_ollama"].dropna().astype(str).unique().tolist()))

    if status_filter != "Todos":
        df_traces = df_traces[df_traces["status"].astype(str) == status_filter]
    if agent_filter != "Todos":
        df_traces = df_traces[df_traces["agentes_texto"].str.contains(agent_filter, na=False, regex=False)]
    if model_filter != "Todos":
        df_traces = df_traces[df_traces["modelo_ollama"].astype(str) == model_filter]

    total_interacoes = len(df_traces)
    total_erros = int((df_traces["status"] == "erro").sum())
    tempo_medio = df_traces["tempo_execucao_ms"].mean()

    taxa_erro = (total_erros / total_interacoes * 100) if total_interacoes else 0
    agentes_unicos = df_traces["agentes"].explode().dropna().nunique()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total de interações", total_interacoes)
    col2.metric("Tempo médio", "N/D" if pd.isna(tempo_medio) else f"{tempo_medio / 1000:.1f} s")
    col3.metric("Erros", total_erros, f"{taxa_erro:.1f}% das interações", delta_color="inverse")
    col4.metric("Agentes em uso", agentes_unicos)

    if df_traces.empty:
        st.info("Nenhum trace encontrado para os filtros selecionados.")
        st.stop()

    chart1, chart2 = st.columns(2)
    with chart1:
        status_counts = df_traces["status"].fillna("indefinido").value_counts().reset_index()
        status_counts.columns = ["status", "quantidade"]
        fig_status = px.pie(status_counts, names="status", values="quantidade", hole=.62, title="Saúde das execuções", color_discrete_sequence=["#00D4FF", "#EF4444", "#7C3AED"])
        fig_status.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#0F172A", margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig_status, use_container_width=True)
    with chart2:
        latency = df_traces.dropna(subset=["tempo_execucao_ms"]).copy()
        latency["execução"] = range(1, len(latency) + 1)
        fig_latency = px.area(latency, x="execução", y="tempo_execucao_ms", title="Latência por execução", color_discrete_sequence=["#7C3AED"])
        fig_latency.update_layout(template="plotly_white", paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF", font_color="#0F172A", margin=dict(l=20, r=20, t=55, b=20))
        st.plotly_chart(fig_latency, use_container_width=True)

    st.markdown('<div class="section-label">Uso dos agentes</div>', unsafe_allow_html=True)

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
    page_header("Agentic intelligence", "Assistente IA", "Converse com agentes especializados conectados aos dados, modelos e ferramentas do laboratório.")

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
        from ai.agentes.ollama_check import (
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
        from ai.agentes.crewai_tools_lab import obter_api_base_url

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
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{
            "role": "assistant",
            "content": "Olá. Sou o assistente de inteligência laboratorial. Posso consultar dados, interpretar métricas e executar previsões com os agentes disponíveis.",
        }]

    history_col, chat_col = st.columns([1, 3])
    selected_prompt = None
    with history_col:
        st.markdown("##### Nova conversa")
        if st.button("＋ Limpar conversa", use_container_width=True):
            st.session_state.chat_messages = st.session_state.chat_messages[:1]
            st.rerun()
        st.markdown("##### Sugestões")
        for indice, exemplo in enumerate(exemplos):
            if st.button(exemplo[:45] + ("…" if len(exemplo) > 45 else ""), key=f"example_{indice}", use_container_width=True):
                selected_prompt = exemplo
        st.caption(f"Modelo ativo: {ollama_model}")
        if CHAT_HISTORY_PATH.exists():
            st.download_button(
                "Baixar histórico",
                data=CHAT_HISTORY_PATH.read_bytes(),
                file_name="assistant_chat_history.jsonl",
                mime="application/jsonl",
                use_container_width=True,
            )

    with chat_col:
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else None):
                st.markdown(message["content"])

    typed_prompt = st.chat_input("Pergunte sobre dados, modelos, métricas ou previsões...")
    pergunta = selected_prompt or typed_prompt
    if pergunta:
        st.session_state.chat_messages.append({"role": "user", "content": pergunta})
        append_chat_history("user", pergunta)
        try:
            from ai.agentes.crewai_agents_lab import executar_crew_lab

            with st.spinner("Orquestrando agentes e ferramentas..."):
                resultado = executar_crew_lab(pergunta)
            resposta = resultado.get("resposta", "Sem resposta.")
            if resultado.get("status") != "ok":
                resposta = f"Não foi possível concluir a execução.\n\n{resposta}"
            st.session_state.chat_messages.append({"role": "assistant", "content": resposta})
            append_chat_history("assistant", resposta)
        except Exception as erro:
            fallback_response = f"Não foi possível carregar a camada de agentes: {erro}"
            st.session_state.chat_messages.append({"role": "assistant", "content": fallback_response})
            append_chat_history("assistant", fallback_response)
        st.rerun()


elif menu == "9. Base de Conhecimento / RAG":
    if area == "📚 Base de Conhecimento":
        page_header("Document intelligence", "Base de Conhecimento", "Gerencie documentos laboratoriais e prepare conteúdo para recuperação semântica.")
    else:
        page_header("Retrieval augmented generation", "RAG Center", "Indexe vetores e encontre conhecimento por similaridade semântica.")

    try:
        rag_docs = carregar_documentos()
    except Exception:
        rag_docs = []
    try:
        rag_vectors = contar_documentos_indexados()
    except Exception:
        rag_vectors = 0
    r1, r2, r3 = st.columns(3)
    r1.metric("Documentos", len(rag_docs))
    r2.metric("Vetores indexados", rag_vectors)
    r3.metric("Última atualização", datetime.now().strftime("%d/%m/%Y"))

    with st.expander("Diagnóstico RAG", expanded=False):
        try:
            status_embedding = verificar_modelo_embedding()
            if status_embedding.get("status") == "ok":
                st.success(status_embedding.get("mensagem"))
            else:
                st.error(status_embedding.get("mensagem"))
                st.code("ollama pull nomic-embed-text", language="powershell")
            st.write("Ollama:", status_embedding.get("base_url"))
            st.write("Modelo de embedding:", status_embedding.get("modelo"))
        except Exception as erro:
            st.error(f"Não foi possível verificar embeddings: {erro}")

        try:
            total_indexado = contar_documentos_indexados()
            st.write("Chunks indexados no ChromaDB:", total_indexado)
        except Exception as erro:
            st.error(str(erro))
            st.code("python -m pip install chromadb", language="powershell")

    arquivos = st.file_uploader(
        "Upload de documentos PDF, DOCX, TXT ou MD",
        type=[extensao.replace(".", "") for extensao in sorted(EXTENSOES_SUPORTADAS)],
        accept_multiple_files=True,
    )

    if arquivos:
        for arquivo in arquivos:
            try:
                caminho = salvar_upload(arquivo)
                st.success(f"Arquivo salvo em data/knowledge_base/: {caminho.name}")
            except Exception as erro:
                st.error(f"Erro ao salvar {arquivo.name}: {erro}")

    try:
        documentos = carregar_documentos()
    except Exception as erro:
        documentos = []
        st.error(f"Não foi possível ler os documentos em data/knowledge_base/: {erro}")
        if "pypdf" in str(erro).lower():
            st.code("python -m pip install pypdf", language="powershell")
        if "python-docx" in str(erro).lower() or "docx" in str(erro).lower():
            st.code("python -m pip install python-docx", language="powershell")

    st.subheader("Documentos disponíveis")
    if documentos:
        doc_cols = st.columns(3)
        for index, documento in enumerate(documentos[:6]):
            with doc_cols[index % 3]:
                tamanho_kb = len(documento.texto.encode("utf-8")) / 1024
                status_doc = "Indexado" if rag_vectors else "Pendente"
                st.markdown(
                    '<div class="model-card">'
                    f'<span class="status-pill">● {status_doc.upper()}</span>'
                    f'<div class="model-name">{documento.nome_arquivo}</div>'
                    f'<div class="model-meta">Texto extraído: {len(documento.texto):,} caracteres<br>Tamanho lógico: {tamanho_kb:.1f} KB<br>Fonte: data/knowledge_base</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        if len(documentos) > 6:
            st.caption(f"Exibindo 6 de {len(documentos)} documentos. A tabela abaixo contém a lista completa.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "arquivo": documento.nome_arquivo,
                        "caminho": documento.caminho,
                        "caracteres_extraidos": len(documento.texto),
                    }
                    for documento in documentos
                ]
            ),
            use_container_width=True,
        )

        st.markdown('<div class="section-label">Gerenciar documentos</div>', unsafe_allow_html=True)
        documentos_por_nome = {documento.nome_arquivo: documento for documento in documentos}
        arquivo_para_excluir = st.selectbox(
            "Arquivo para excluir da Base de Conhecimento",
            list(documentos_por_nome.keys()),
            key="rag_delete_file_select",
        )
        confirmar_exclusao = st.checkbox(
            f"Confirmo que desejo desativar {arquivo_para_excluir}, movê-lo para "
            "data/documentos_inativos e remover seus vetores.",
            key="rag_delete_file_confirm",
        )
        if st.button(
            "Excluir documento selecionado",
            disabled=not confirmar_exclusao,
            type="secondary",
        ):
            try:
                mensagem_exclusao = delete_knowledge_document(documentos_por_nome[arquivo_para_excluir])
                st.success(mensagem_exclusao)
                st.rerun()
            except Exception as erro:
                st.error(f"Não foi possível excluir o documento: {erro}")
    else:
        st.info("Nenhum documento encontrado em data/knowledge_base/.")

    if st.button("Indexar documentos", type="primary"):
        if not documentos:
            st.warning("Envie ou adicione documentos antes de indexar.")
        else:
            try:
                with st.spinner("Indexando documentos com embeddings do Ollama..."):
                    resultado_indexacao = indexar_documentos()
                st.success(
                    "Indexação concluída: "
                    f"{resultado_indexacao['arquivos']} arquivo(s), "
                    f"{resultado_indexacao['chunks']} chunk(s)."
                )
                st.write("Modelo de embedding:", resultado_indexacao["modelo_embedding"])
                st.dataframe(
                    pd.DataFrame(resultado_indexacao["detalhes"]),
                    use_container_width=True,
                )
            except Exception as erro:
                st.error(str(erro))
                if "ChromaDB" in str(erro) or "chromadb" in str(erro).lower():
                    st.code("python -m pip install chromadb", language="powershell")
                if "embedding" in str(erro).lower() or "nomic" in str(erro).lower():
                    st.code("ollama pull nomic-embed-text", language="powershell")

    st.divider()
    st.subheader("Buscar na base")
    pergunta_rag = st.text_area(
        "Pergunta",
        height=100,
        placeholder="Quais foram as dificuldades encontradas na Etapa 2?",
    )
    top_k = st.slider("Quantidade de trechos", min_value=1, max_value=10, value=4)

    if st.button("Buscar na base"):
        if not pergunta_rag.strip():
            st.warning("Digite uma pergunta antes de buscar.")
        else:
            try:
                if classificar_pergunta(pergunta_rag) == "numerica":
                    with st.spinner("Consultando DuckDB para pergunta numérica..."):
                        resultado_numerico = responder_pergunta_numerica(pergunta_rag)
                    st.success("Pergunta numérica respondida via DuckDB, sem Ollama e sem RAG textual.")
                    st.write(formatar_resposta_numerica(resultado_numerico))
                    with st.expander("Resultado técnico"):
                        st.json(resultado_numerico)
                    st.stop()

                with st.spinner("Buscando trechos semanticamente próximos..."):
                    trechos = buscar_trechos(pergunta_rag, k=top_k)
                if not trechos:
                    st.warning(
                        "Nenhum trecho encontrado. Verifique se os documentos foram indexados."
                    )
                else:
                    for indice, trecho in enumerate(trechos, start=1):
                        score = trecho.get("score")
                        score_texto = f"{score:.3f}" if isinstance(score, float) else "N/D"
                        with st.expander(
                            f"Trecho {indice} | {trecho['fonte']} | "
                            f"chunk {trecho['chunk']} | score {score_texto}",
                            expanded=indice == 1,
                        ):
                            st.write(trecho["texto"])
                            st.caption(f"Fonte: {trecho['fonte']}")
                            st.caption(f"Caminho: {trecho.get('caminho', '')}")
                            st.caption(f"Distância: {trecho.get('distancia')}")
            except Exception as erro:
                st.error(str(erro))
                st.code(
                    "python -m pip install chromadb\nollama pull nomic-embed-text",
                    language="powershell",
                )


elif menu == "10. Configurações":
    page_header("Platform administration", "Configurações", "Visualize integrações, runtime local e parâmetros da plataforma.")

    env_values = {}
    env_path = BASE_DIR / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.split("=", 1)
                env_values[key.strip()] = value.strip()

    c1, c2, c3 = st.columns(3)
    c1.markdown('<div class="model-card"><span class="status-pill">● CONNECTED</span><div class="model-name">DuckDB</div><div class="model-meta">Banco analítico local<br>Persistência: data/lab_ia.duckdb</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="model-card"><span class="status-pill">● CONFIGURED</span><div class="model-name">Ollama Runtime</div><div class="model-meta">Endpoint: {env_values.get("OLLAMA_BASE_URL", "http://localhost:11434")}<br>Modelo: {env_values.get("OLLAMA_MODEL", "não definido")}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="model-card"><span class="status-pill">● CONNECTED</span><div class="model-name">FastAPI</div><div class="model-meta">Endpoint: {env_values.get("API_BASE_URL", "http://localhost:8000")}<br>Serviço de inferência</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Preferências da plataforma</div>', unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        st.write("Tema ativo:", theme_mode)
        st.write("Escala de fonte:", f"{font_scale:.2f}x")
        st.selectbox("Idioma", ["Português (Brasil)"])
        st.toggle("Animações e microinterações", value=True)
    with right:
        st.selectbox("Runtime de IA", ["Ollama local"])
        st.selectbox("Motor vetorial", ["ChromaDB local"])
        st.toggle("Registrar traces de agentes", value=True)
    st.info("As configurações exibidas refletem o ambiente local atual. Alterações persistentes continuam sendo administradas pelo arquivo .env.")

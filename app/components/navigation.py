import streamlit as st


def navigate(area_value: str, data_module: str | None = None) -> None:
    st.session_state["active_area"] = area_value
    if data_module:
        st.session_state["active_data_module"] = data_module
    st.rerun()

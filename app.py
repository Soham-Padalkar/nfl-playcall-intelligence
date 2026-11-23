"""Run with streamlit run app.py."""

import streamlit as st

from playcall import ui, ui_evidence, ui_forecast, ui_tools

PAGES = {
    "Matchup Overview": ui_evidence.overview,
    "Opening-Drive Predictor": ui_forecast.predictor,
    "Playcaller Matchup Explorer": ui_evidence.matchups,
    "Defensive Similarity Explorer": ui_evidence.similarity,
    "Historical Tendencies": ui_evidence.history,
    "Video Analyzer": ui_tools.video,
    "Model Laboratory": ui_tools.laboratory,
    "Live Prediction & Evaluation": ui_forecast.live,
    "GPU & System Monitor": ui_tools.monitor,
}

st.set_page_config(page_title="NFL Playcall Intelligence", page_icon="🏈", layout="wide")
ui.style()
with st.sidebar:
    st.markdown("**NFL PLAYCALL INTELLIGENCE**")
    st.caption("ATL  /  MIN")
    page = st.radio("Dashboard pages", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("Research workspace · Local data and forecasts")

st.title(page)
PAGES[page]()

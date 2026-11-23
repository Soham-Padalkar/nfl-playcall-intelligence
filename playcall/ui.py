"""Shared dashboard data access and chart styling."""

import json
from pathlib import Path
from pickle import UnpicklingError

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from playcall.data import DataStore

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODEL = ROOT / "models" / "playcall.joblib"
REPORT = ROOT / "models" / "backtest.json"
CONTEXT = Path(__file__).with_name("context.json")
RUN = "#84eac6"
PASS = "#89afff"
TEXT = "#e8eef7"
GRAY = "#93a5bd"


def style():
    st.markdown(
        """
    <style>
    .block-container {max-width: 1360px; padding-top: 3.2rem; padding-bottom: 3rem;}
    [data-testid="stSidebarContent"] {border-right: 1px solid #27374c;}
    [data-testid="stSidebar"] [role="radiogroup"] {gap: 0.4rem;}
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {color: #b8c8dd;}
    [data-testid="stMetric"] {
      background: #141e2e; border: 1px solid #2a3a50; border-radius: 12px;
      padding: 1rem 1.05rem;
    }
    h1 {letter-spacing: -0.035em !important;}
    h2, h3 {letter-spacing: -0.02em !important;}
    .matchup-banner {
      background: linear-gradient(125deg,#2a1727 0%,#152132 45%,#251e42 100%);
      border: 1px solid #34465f; border-radius: 16px; padding: 1.5rem 1.65rem;
    }
    .matchup-banner small {color:#bfcde0; font-weight:600; letter-spacing:.09em;}
    .matchup-banner strong {display:block; font-size:clamp(1.4rem,4vw,2.7rem);
      letter-spacing:-.035em; margin:.16rem 0 .3rem;}
    .matchup-banner span {color:#d5ddec;}
    </style>
    """,
        unsafe_allow_html=True,
    )


def context():
    return json.loads(CONTEXT.read_text(encoding="utf-8"))


def get_manifest():
    return DataStore(DATA).manifest()


@st.cache_data(show_spinner=False, max_entries=4)
def _target_schedule(root, dataset_version, game_id):
    schedules = DataStore(root).schedules()
    if schedules.empty:
        return {}
    match = schedules.loc[schedules["game_id"].eq(game_id)]
    return match.iloc[0].to_dict() if not match.empty else {}


def target_schedule(game_id):
    return _target_schedule(DATA, get_manifest()["dataset_version"], game_id)


def scheduled_rest(game_id):
    game = target_schedule(game_id)
    days = (game.get("away_rest"), game.get("home_rest"))
    return tuple(int(value) if pd.notna(value) and 0 <= value <= 24 else None for value in days)


@st.cache_data(show_spinner="Reading cached plays…", max_entries=2)
def _plays(root, dataset_version):
    return DataStore(root).plays()


def get_plays():
    return _plays(DATA, get_manifest()["dataset_version"])


@st.cache_resource(show_spinner="Loading the playcall model…", max_entries=2)
def _model(path, signature):
    from playcall.models import load_models

    return load_models(path)


def get_model():
    if MODEL.is_file():
        try:
            return _model(MODEL, MODEL.stat().st_mtime_ns)
        except (
            AttributeError,
            EOFError,
            ImportError,
            OSError,
            UnpicklingError,
            ValueError,
        ) as error:
            st.warning(
                f"The local model checkpoint could not be loaded. Train the model again. {error}"
            )
    return None


def require_model():
    model = get_model()
    if model is None:
        st.info(
            "Train a model from the local datasets to enable this page. See the README for the training command."
        )
    return model


def figure(fig, height=360):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        paper_bgcolor="#0c121d",
        plot_bgcolor="#101927",
        font=dict(color=TEXT, family="Arial"),
        margin=dict(l=16, r=16, b=20, t=40),
        legend=dict(orientation="h", y=1.10),
    )
    fig.update_xaxes(gridcolor="#26354a")
    fig.update_yaxes(gridcolor="#26354a")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})


def reliability(rows, name="Model"):
    """Rows contain the forecast mean, observed pass share and bin count."""
    fig = go.Figure()
    fig.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line=dict(color=GRAY, dash="dot"),
    )
    if len(rows):
        frame = pd.DataFrame(rows)
        fig.add_scatter(
            x=frame["forecast"],
            y=frame["observed"],
            customdata=frame["n"],
            mode="markers",
            name=name,
            marker=dict(color=RUN, size=11),
            hovertemplate="Forecast %{x:.1%}<br>Observed %{y:.1%}<br>%{customdata} plays<extra></extra>",
        )
    fig.update_xaxes(title="Mean predicted pass probability", tickformat=".0%", range=[0, 1])
    fig.update_yaxes(title="Observed pass share", tickformat=".0%", range=[0, 1])
    figure(fig)


def show_prediction(prediction):
    cols = st.columns(2)
    cols[0].metric("Run probability", f"{prediction['p_run']:.1%}")
    cols[1].metric("Pass probability", f"{prediction['p_pass']:.1%}")
    if prediction.get("selected_model"):
        st.caption(f"Selected model · {prediction['selected_model']}")
    uncertainty = prediction.get("uncertainty", {})
    interval = uncertainty.get("wilson95")
    if interval:
        st.caption(
            f"Historical reference · {uncertainty['n']:,} plays, {uncertainty['basis']} · "
            f"95% interval for observed pass share {interval[0]:.1%}–{interval[1]:.1%}"
        )
    tactical = prediction.get("tactical", {})
    if tactical:
        with st.expander("Tactical estimates from the historical reference"):
            rows = []
            for distribution in tactical.values():
                labels = distribution["probabilities"]
                estimate = max(labels, key=labels.get)
                rows.append(
                    {
                        "Attribute": distribution["label"],
                        "Most frequent": estimate,
                        "Share of labeled plays": labels[estimate],
                        "Labeled plays": distribution["n"],
                        "Label coverage": distribution["coverage"],
                    }
                )
            st.dataframe(
                rows,
                width="stretch",
                hide_index=True,
                column_config={
                    key: st.column_config.NumberColumn(key, format="percent")
                    for key in ("Share of labeled plays", "Label coverage")
                },
            )
            st.caption(
                "These estimates use labeled historical plays. Exact play concepts need verified film labels."
            )
    if prediction.get("components"):
        with st.expander("Ensemble components"):
            show_components(prediction)


def show_components(prediction):
    labels = {
        "A": "A · Direct Flores matchups",
        "B": "B · Comparable defenses",
        "C": "C · Recent tendencies",
    }
    rows = [
        {
            "Component": labels.get(key, key),
            "Pass estimate": item["p_pass"],
            "Effective sample": item["effective_n"],
            "Ensemble weight": item["weight"],
            "Sample basis": item.get("sample_basis"),
        }
        for key, item in prediction["components"].items()
    ]
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={
            key: st.column_config.NumberColumn(key, format="percent")
            for key in ("Pass estimate", "Ensemble weight")
        },
    )
    st.caption(
        "Weights apply to the ensemble estimate. The selected model may use a different method."
    )


def source_table(manifest):
    rows = [
        {
            "Source": name,
            "Status": entry.get("status"),
            "Rows": entry.get("rows"),
            "Retrieved": entry.get("retrieved_at"),
            "Address": entry.get("url"),
        }
        for name, entry in manifest["sources"].items()
    ]
    if rows:
        st.dataframe(
            rows,
            hide_index=True,
            width="stretch",
            column_config={
                "Address": st.column_config.LinkColumn("Address", display_text="Open source")
            },
        )
    else:
        st.info("This installation has no source metadata yet.")

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from playcall import ui, ui_forecast
from playcall.neural_selection import FIELDS, Winner
from playcall.presnap import PERSONNEL_WEIGHT, ObservedModel, covered, features, with_game_history


def test_observed_alignment_updates_probability_and_missing_values_fall_back():
    base = {"down": 1, "ydstogo": 10, "yardline_100": 70}
    train = features(
        pd.DataFrame(
            [{**base, "qb_location": "S", "target": True}] * 80
            + [{**base, "qb_location": "U", "target": False}] * 80
        )
    )
    model = ObservedModel(train, ("qb_location",), 12)
    queries = features(pd.DataFrame([{**base, "qb_location": k} for k in ("S", "U", None, "X")]))
    p = model.predict(queries)
    assert p[0] > 0.85
    assert p[1] < 0.15
    assert p[2:] == pytest.approx([0.5, 0.5])


def test_confidence_group_requires_complete_observations():
    plays = pd.DataFrame(
        {"target": [True, True, False], "qb_location": ["S", None, "U"], "game_id": ["a", "b", "c"]}
    )
    result = covered(plays, np.array([0.93, 0.99, 0.40]), 0.85, ("qb_location",))
    assert result["n"] == 1
    assert result["coverage"] == pytest.approx(1 / 3)
    assert result["game_bootstrap_lower_95"] is None


def test_history_uses_only_earlier_calls_by_the_same_offense_and_game():
    rows = [
        ("a", 1, "ATL", "run"),
        ("a", 2, "MIN", "pass"),
        ("a", 3, "ATL", "pass"),
        ("a", 4, "ATL", "run"),
        ("a", 5, "ATL", "pass"),
        ("b", 1, "ATL", "pass"),
    ]
    plays = pd.DataFrame(rows, columns=["game_id", "play_id", "posteam", "call"]).iloc[::-1]
    result = with_game_history(plays).set_index(["game_id", "play_id"])
    assert pd.isna(result.loc[("a", 1), "recent_same_game"])
    assert pd.isna(result.loc[("a", 2), "recent_same_game"])
    assert result.loc[("a", 5), "recent_same_game"] == "RPR"
    assert result.loc[("a", 5), "previous_same_game"] == "R"
    assert pd.isna(result.loc[("b", 1), "recent_same_game"])


def test_personnel_gets_more_weight_and_accepts_charted_position_counts():
    base = {"down": 1, "ydstogo": 10, "yardline_100": 75, "qb_location": "S"}
    raw_offense = "1 C, 2 G, 1 QB, 1 RB, 2 T, 1 TE, 3 WR"
    raw_defense = "3 CB, 2 DE, 2 DT, 1 FS, 1 MLB, 1 OLB, 1 SS"
    train = features(
        pd.DataFrame(
            [{**base, "offense_personnel": raw_offense, "target": True}] * 40
            + [{**base, "offense_personnel": "12", "target": False}] * 40
        )
    )
    fields = ("qb_location", "offense_personnel")
    query = features(
        pd.DataFrame([{**base, "offense_personnel": "11", "defense_personnel": raw_defense}])
    )
    assert query.iloc[0]["defense_personnel"] == "4-2-5"
    assert (
        ObservedModel(train, fields, 32, PERSONNEL_WEIGHT).predict(query)[0]
        > (ObservedModel(train, fields, 32).predict(query)[0])
    )


def test_dashboard_uses_observed_alignment(monkeypatch, tmp_path):
    def predict(model_input):
        row = model_input.iloc[0]
        assert row["qb_location"] == "S"
        assert row["offense_personnel"] == "11"
        assert row["recent_same_game"] == "RPP"
        return np.array([0.88])

    monkeypatch.setattr(ui, "MODEL", tmp_path / "playcall.joblib")
    (tmp_path / "neural-winner.json").touch()
    monkeypatch.setattr(
        ui_forecast, "_load_observed_model", lambda *_: SimpleNamespace(predict=predict, name="MLP")
    )
    app = AppTest.from_string(
        """
from types import SimpleNamespace
from playcall.ui_forecast import _observed_prediction
_observed_prediction(
    SimpleNamespace(down=1, distance=10, yardline_100=75,
                    game_seconds_remaining=3550, score_differential=0), ["run", "pass", "pass"]
)
"""
    ).run()
    assert not app.exception
    next(widget for widget in app.selectbox if widget.label == "Quarterback alignment").set_value(
        "S"
    )
    next(widget for widget in app.text_input if widget.label == "Offensive personnel").set_value(
        "11"
    )
    app.button[0].click().run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Pass probability"] == "88.0%"


def test_exported_classifier_handles_a_first_snap_without_setup():
    weights = {
        "first.weight": [[0.0] * 11],
        "first.bias": [1.0],
        "second.weight": [[2.0]],
        "second.bias": [0.0],
        "out.weight": [[1.0]],
        "out.bias": [0.0],
        "gate.weight": [[0.0]],
        "gate.bias": [0.0],
    }
    winner = Winner(
        {
            "selected": "Gated MLP",
            "version": "fixture",
            "weights": weights,
            "vocab": {key: [] for key in FIELDS},
        }
    )
    assert winner.predict_state({}) == pytest.approx(0.73105858)

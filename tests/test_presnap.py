from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from playcall import ui, ui_forecast
from playcall.presnap import ObservedModel, covered, features


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


def test_dashboard_uses_observed_alignment(monkeypatch):
    def predict(model_input):
        assert model_input.iloc[0]["qb_location"] == "S"
        return np.array([0.88])

    monkeypatch.setattr(ui, "get_manifest", lambda: {"dataset_version": "test"})
    monkeypatch.setattr(
        ui_forecast, "_load_observed_model", lambda *_: SimpleNamespace(predict=predict)
    )
    app = AppTest.from_string(
        """
from types import SimpleNamespace
from playcall.ui_forecast import _observed_prediction
_observed_prediction(SimpleNamespace(down=1, distance=10, yardline_100=75))
"""
    ).run()
    assert not app.exception
    next(widget for widget in app.selectbox if widget.label == "Quarterback alignment").set_value(
        "S"
    )
    app.button[0].click().run()
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["Pass probability"] == "88.0%"

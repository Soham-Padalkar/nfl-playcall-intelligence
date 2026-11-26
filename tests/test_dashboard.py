"""Dashboard smoke tests using isolated local assets and forecast records."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

from streamlit.testing.v1 import AppTest

from playcall import ui, ui_forecast
from playcall.simulation import list_forecasts
from playcall.ui_tools import _snap_alignment


class TestForecastModel:
    __test__ = False
    version = "test-model"
    dataset_version = "test-data"
    report = {"git_commit": "snapshot-test"}

    def predict(self, state):
        pass_share = 0.6 if state.down == 1 else 0.75
        prediction = {"p_pass": pass_share, "p_run": 1 - pass_share, "selected_model": "test"}
        return {**prediction, "setups": self.predict_setups(state, prediction)}

    call_probabilities = predict

    def predict_setups(self, state, prediction):
        return {
            "pass": {
                "options": [
                    {
                        "personnel": "12" if state.down == 1 else "11",
                        "formation": "Shotgun",
                        "concept": "Play-action pass" if state.down == 1 else "Short pass left",
                        "weighted_share": prediction["p_pass"],
                        "conditional_share": 1.0,
                        "samples": 1,
                    }
                ]
            }
        }

    def outcome_distribution(self, state, call):
        return [
            {
                "probability": 1.0,
                "yards": 3 if call == "run" else 8,
                "turnover": False,
                "touchdown": False,
                "safety": False,
            }
        ]


def isolated_app(monkeypatch, tmp_path):
    monkeypatch.setattr(ui, "DATA", tmp_path / "data")
    monkeypatch.setattr(ui, "MODEL", tmp_path / "model.joblib")
    monkeypatch.setattr(ui, "REPORT", tmp_path / "backtest.json")
    monkeypatch.setattr(ui_forecast, "PREDICTIONS", tmp_path / "predictions")
    monkeypatch.setattr("playcall.gpu.telemetry", lambda: {"cuda_available": False})
    return AppTest.from_file(str(Path(ui.ROOT) / "app.py"), default_timeout=15).run()


def test_dashboard_pages_without_optional_assets(monkeypatch, tmp_path):
    app = isolated_app(monkeypatch, tmp_path)
    assert not app.exception
    assert len(app.sidebar.radio[0].options) == 9
    for page in list(app.sidebar.radio[0].options)[1:]:
        app.sidebar.radio[0].set_value(page).run()
        assert not app.exception, page
        assert app.title[0].value == page


def test_freeze_and_adaptive_entry(monkeypatch, tmp_path):
    app = isolated_app(monkeypatch, tmp_path)
    game = {
        "game_id": "test-game",
        "kickoff_utc": (datetime.now(UTC) + timedelta(days=1)).isoformat(),
    }
    monkeypatch.setattr(ui, "context", lambda: {"game": game})
    monkeypatch.setattr(ui, "get_model", TestForecastModel)
    monkeypatch.setattr(ui, "target_schedule", lambda game_id: {"away_rest": 14, "home_rest": 7})
    app.sidebar.radio[0].set_value("Opening-Drive Predictor").run()
    assert not app.exception
    rest = {field.label: field.value for field in app.number_input if "days of rest" in field.label}
    assert rest == {"Atlanta days of rest": 14, "Minnesota days of rest": 7}
    branch = next(box for box in app.selectbox if box.label == "Show a branch")
    assert len(branch.options) > 1
    branch.select_index(1).run()
    assert not app.exception
    next(button for button in app.button if button.label == "Freeze pregame forecast").click().run()
    assert not app.exception
    forecasts = list_forecasts(ui_forecast.PREDICTIONS, game_id="test-game", kind="pregame")
    assert len(forecasts) == 1
    assert forecasts[0]["forecast"]["root"]["prediction"]["p_pass"] == 0.6
    assert forecasts[0]["state"]["offense_rest"] == 14
    assert forecasts[0]["state"]["defense_rest"] == 7

    # Exercise the post-kickoff UI while preserving the frozen record.
    game["kickoff_utc"] = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    next(button for button in app.button if button.label == "Open live play log").click().run()
    assert not app.exception
    next(field for field in app.number_input if field.label == "Yards gained").set_value(3)
    next(button for button in app.button if button.label == "Record play and update").click().run()
    assert not app.exception
    records = list_forecasts(ui_forecast.PREDICTIONS, game_id="test-game", kind="adaptive")
    assert len(records) == 1
    assert records[0]["parent_id"] == forecasts[0]["id"]
    assert records[0]["actual"]["yards"] == 3
    assert records[0]["state"]["down"] == 2
    assert records[0]["state"]["distance"] == 7
    assert records[0]["forecast"]["root"]["prediction"]["p_pass"] == 0.75
    assert any("Short pass left" in entry.value for entry in app.markdown)

    next(field for field in app.number_input if field.label == "Yards gained").set_value(2)
    for label, value in (
        ("Official next down", 1),
        ("Official yards to gain", 10),
        ("Official yards to goal", 65),
    ):
        next(field for field in app.number_input if field.label == label).set_value(value)
    next(button for button in app.button if button.label == "Record play and update").click().run()
    assert not app.exception
    records = list_forecasts(ui_forecast.PREDICTIONS, game_id="test-game", kind="adaptive")
    assert len(records) == 2
    assert records[1]["parent_id"] == records[0]["id"]
    assert records[1]["state"]["down"] == 1
    assert records[1]["state"]["distance"] == 10
    assert records[1]["state"]["yardline_100"] == 65
    evaluation = ui_forecast._series(records, forecasts[0])
    assert evaluation.loc[0, "Frozen"] == 0.6
    assert evaluation.loc[1, "Adaptive"] == 0.75
    assert len(list_forecasts(ui_forecast.PREDICTIONS, kind="pregame")) == 1


def test_alignment_respects_manual_evidence():
    assert "Under-center" in _snap_alignment(0.5, 0.25, "Unknown")
    assert "Pistol-like" in _snap_alignment(3.5, 0.5, "Directly behind")
    assert "Shotgun-like" in _snap_alignment(5.5, 0.5, "Offset")
    assert "Ambiguous" in _snap_alignment(4.5, 1.0, "Directly behind")
    assert "Unknown" in _snap_alignment(5.5, None, "Offset")


def test_projected_drive_ignores_a_high_mass_path_pruned_before_the_horizon():
    def node(mass, children=(), ended=False):
        return {"mass": mass, "children": list(children), "state": {"drive_ended": ended}}

    pruned = {"call": "run", "node": node(0.5)}
    continuation = {"call": "pass", "node": node(0.2)}
    completed = {"call": "run", "node": node(0.3, [continuation])}
    mass, path = ui_forecast._likely_path(node(1, [pruned, completed]), 2)
    assert mass == 0.2
    assert [edge["call"] for _, edge in path] == ["run", "pass"]

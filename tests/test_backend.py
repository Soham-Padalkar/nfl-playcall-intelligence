from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from playcall.data import DataStore, validate_parquet
from playcall.models import (
    _ensemble,
    _similarity,
    assign_callers,
    chronological_split,
    load_context,
)
from playcall.simulation import (
    GameState,
    advance_state,
    freeze_forecast,
    list_forecasts,
    opening_tree,
    record_adaptive,
)


def test_parquet_ingestion_and_opening_drive(tmp_path):
    store = DataStore(tmp_path)
    store.raw.mkdir()
    common = {
        "game_id": "2025_01_ATL_MIN",
        "season": 2025,
        "week": 1,
        "game_date": "2025-09-05",
        "posteam": "ATL",
        "defteam": "MIN",
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 75,
        "game_seconds_remaining": 3600,
        "score_differential": 0,
        "yards_gained": 4,
    }
    rows = [
        {**common, "play_id": 11, "play_type": "run", "drive": 1, "qb_dropback": 1},
        {**common, "play_id": 22, "play_type": "run", "drive": 1, "qb_dropback": 0},
        {**common, "play_id": 22, "play_type": "run", "drive": 1, "qb_dropback": 0},
        {**common, "play_id": 33, "play_type": "run", "drive": 3, "qb_dropback": 0},
    ]
    path = store.raw / "pbp_2025.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    assert validate_parquet(path, set(common)) == 4
    with pytest.raises(ValueError, match="missing columns"):
        validate_parquet(path, {"field_hash_that_is_absent"})
    plays = store.plays()
    assert plays["opening_drive"].tolist() == [True, True, False]
    assert plays["call"].tolist() == ["pass", "run", "run"]
    assert pd.isna(plays["previous_call"].iloc[0])
    assert plays["previous_call"].iloc[1] == "pass"


def test_coach_intervals_exclude_unverified_games():
    frame = pd.DataFrame(
        {
            "posteam": ["CLE", "CLE", "ATL", "ATL"],
            "season": [2025, 2025, 2025, 2026],
            "week": [5, 10, 12, 1],
        }
    )
    assert assign_callers(frame, load_context()).tolist() == [
        "Unknown",
        "Tommy Rees",
        "Unknown",
        "Tommy Rees",
    ]


def test_similarity_excludes_target_from_model_b():
    from playcall.models import PROFILE_FEATURES

    frame = pd.DataFrame(
        np.repeat([[0.0], [0.2], [1.0]], len(PROFILE_FEATURES), axis=1),
        columns=PROFILE_FEATURES,
        index=["MIN", "ATL", "NYJ"],
    )
    weights = _similarity(frame)
    assert weights.loc["MIN", "MIN"] == 0
    assert weights.loc["MIN", "ATL"] > weights.loc["MIN", "NYJ"] > 0


def test_chronological_split_keeps_seasons_intact():
    frame = pd.DataFrame({"season": [2026, 2022, 2023, 2025, 2024, 2025]})
    training, validation, test = chronological_split(frame)
    assert training["season"].tolist() == [2022, 2023]
    assert validation["season"].tolist() == [2024]
    assert set(test["season"]) == {2025, 2026}


def test_ensemble_ignores_unsupported_components():
    rates = {
        key: np.array([value])
        for key, value in {
            "A": 0.9,
            "B": 0.8,
            "C": 0.4,
            "A_n": 0,
            "B_n": 0,
        }.items()
    }
    probability, weights = _ensemble(rates, 1.0)
    assert probability[0] == 0.4
    assert [weight[0] for weight in weights] == [0, 0, 1]


class SmallModel:
    version = "test"
    dataset_version = "test"
    report = {"git_commit": "test"}

    def predict(self, state):
        return self.call_probabilities(state)

    def call_probabilities(self, state):
        return {"p_pass": 0.4, "p_run": 0.6}

    def outcome_distribution(self, state, call):
        return [
            {
                "yards": 4,
                "turnover": False,
                "touchdown": False,
                "safety": False,
                "probability": 0.7,
            },
            {
                "yards": 15,
                "turnover": False,
                "touchdown": False,
                "safety": False,
                "probability": 0.3,
            },
        ]


def test_tree_preserves_global_mass():
    tree = opening_tree(SmallModel(), GameState(), depth=2, beam_width=1)
    assert tree["retained_mass"] + tree["pruned_mass"] == pytest.approx(1.0)
    assert tree["retained_mass"] == pytest.approx(0.42**2)
    first = tree["root"]["children"][0]
    assert first["call"] == "run"
    assert first["node"]["state"]["down"] == 2
    assert first["node"]["state"]["distance"] == 6


def test_state_transitions():
    converted = advance_state(GameState(down=3, distance=6, yardline_100=14), "run", 8)
    assert (converted.down, converted.distance, converted.yardline_100) == (1, 6, 6)
    assert advance_state(GameState(down=4), "pass", 0).end_reason == "turnover on downs"
    assert advance_state(GameState(yardline_100=2), "run", 2).score_differential == 6
    assert advance_state(GameState(yardline_100=99), "run", -1).end_reason == "safety"
    assert advance_state(GameState(), "pass", 0, turnover=True).end_reason == "turnover"


def test_frozen_and_adaptive_records(tmp_path):
    model = SmallModel()
    state = GameState()
    kickoff = datetime.now(UTC) + timedelta(hours=2)
    first = freeze_forecast(model, state, "fixture", kickoff, tmp_path, depth=1)
    frozen_path = Path(tmp_path, "pregame", f"{first['id']}.json")
    original = frozen_path.read_bytes()
    adaptive = record_adaptive(
        model, state, "run", 7, "fixture", tmp_path, parent_id=first["id"], depth=1
    )
    assert adaptive["state"]["distance"] == 3
    assert adaptive["prediction_before"] == first["forecast"]["root"]["prediction"]
    assert adaptive["forecast"]["root"]["state"] == adaptive["state"]
    assert len(list_forecasts(tmp_path, "fixture")) == 2
    assert frozen_path.read_bytes() == original
    with pytest.raises(ValueError, match="future"):
        freeze_forecast(model, state, "fixture", datetime.now(UTC), tmp_path, depth=1)

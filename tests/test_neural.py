"""Offline coverage for the optional neural experiment."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from playcall import neural  # noqa: E402


@pytest.fixture
def plays():
    rows = []
    for year in range(2022, 2025):
        for drive, team in enumerate(("NE", "BUF"), start=1):
            for play, call in enumerate(("run", "pass", "run"), start=1):
                rows.append(
                    {
                        "game_date": f"{year}-09-15",
                        "season": year,
                        "game_id": f"{year}_BUF_NE",
                        "posteam": team,
                        "defteam": "BUF" if team == "NE" else "NE",
                        "drive": drive,
                        "opening_drive": True,
                        "play_id": 10 * drive + play,
                        "call": call,
                        "down": play,
                        "ydstogo": 10 - play,
                        "yardline_100": 75 - play,
                        "score_differential": 0,
                        "game_seconds_remaining": 3500 - drive * play,
                    }
                )
    return pd.DataFrame(rows).sample(frac=1, random_state=12)


def test_temporal_encoding_uses_earlier_calls_in_same_drive(plays):
    splits = neural.opening_splits(plays, "2024-09-15")
    _, datasets = neural.encode_splits(splits, 2)
    for features, history, labels in (dataset.tensors for dataset in datasets):
        assert history[0].count_nonzero().item() == history[3].count_nonzero().item() == 0
        assert history[1, -1, -2:].tolist() == [0, 1]
        assert history[2, -1, -2:].tolist() == [1, 1]
        assert torch.equal(features[1], history[2, -1, :-2])
        assert labels.tolist() == [0, 1, 0, 0, 1, 0]


def test_split_rejects_overlapping_season_dates(plays):
    plays.loc[plays["season"].eq(2022), "game_date"] = "2023-09-15"
    with pytest.raises(ValueError, match="overlap"):
        neural.opening_splits(plays, "2024-09-15")


def test_cpu_experiment_produces_metrics_and_safe_checkpoints(plays, tmp_path):
    config = neural.TrainingConfig(epochs=2, batch_size=2, accumulation_steps=2, hidden_size=4)
    report = neural.experiment(
        plays, "2024-09-15", "test-data", config, device="cpu", checkpoints=tmp_path
    )
    assert report["splits"]["test"]["opening_drives"] == 2
    for model in ("league_mean", "logistic", "mlp", "gru"):
        assert np.isfinite(report["metrics"][model]["opening_drives"]["log_loss"])
        assert report["metrics"][model]["first_eligible_play"]["n"] == 2
    for model in ("mlp", "gru"):
        checkpoint = torch.load(tmp_path / f"{model}.pt", weights_only=True)
        assert checkpoint["model"] == model
        assert checkpoint["dataset_version"] == "test-data"
        assert report["metrics"][model]["amp_dtype"] is None


def test_oom_retry_keeps_accumulation_then_uses_cpu(monkeypatch):
    attempts = []

    def mocked_train(kind, data, config, device):
        attempts.append((device.type, config.batch_size, config.accumulation_steps))
        if device.type == "cuda":
            raise torch.cuda.OutOfMemoryError()
        return None, None, {"device": device.type}

    monkeypatch.setattr(neural, "_train_candidate", mocked_train)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    config = neural.TrainingConfig(batch_size=4, accumulation_steps=2)
    _, _, details = neural._train_with_fallback("mlp", None, config, torch.device("cuda"))
    assert attempts == [("cuda", 4, 2), ("cuda", 2, 4), ("cuda", 1, 8), ("cpu", 4, 2)]
    assert len(details["oom_retries"]) == 3

"""Run/pass probabilities when the operator can see the setup for the current snap."""

import argparse
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from playcall.data import DataStore, _atomic_json

CUTOFF = "2026-11-22"
OBSERVABLES = {
    "qb_location": "Quarterback alignment",
    "n_defense_box": "Defenders in the box",
    "n_offense_backfield": "Players in the offensive backfield",
    "starting_hash": "Field hash",
    "offense_personnel": "Offensive personnel",
    "defense_personnel": "Defensive personnel",
    "offense_formation": "Offensive formation",
    "is_motion": "Motion at or before the snap",
}
SITUATION = ("down", "distance_band", "field_band")
CORE = ("qb_location", "n_defense_box", "n_offense_backfield", "starting_hash")
GROUPS = ("offense_formation", "offense_personnel", "defense_personnel")
CANDIDATES = {
    "Situation only": (),
    "QB alignment": ("qb_location",),
    "Alignment and box": CORE[:2],
    "Alignment, box and backfield": CORE[:3],
    "Alignment, box, backfield and motion": (*CORE[:3], "is_motion"),
    "Alignment, box, backfield and hash": CORE,
    "Formation and personnel": (CORE[0], *GROUPS),
    "All observable fields": (*CORE, *GROUPS, "is_motion"),
}


def features(frame):
    data = frame.assign(
        distance_band=np.digitize(frame["ydstogo"], [3, 7, 11, 16]),
        field_band=np.digitize(frame["yardline_100"], [21, 51, 81]),
    )
    for field in OBSERVABLES:
        values = data.get(field, pd.Series(pd.NA, index=data.index))
        if field in CORE[1:3]:
            values = pd.to_numeric(values, errors="coerce").astype("Int64")
        data[field] = values.astype("string").str.strip().replace("", pd.NA)
    return data


class ObservedModel:
    """Shrink each count toward its parent estimate. Missing inputs skip that update."""

    def __init__(self, train, fields, strength):
        self.fields = fields
        self.strength = strength
        self.mean = float(train["target"].mean())
        groups = [SITUATION + fields[:index] for index in range(len(fields) + 1)]
        # Keep separate one-field fallback tables for setups with missing cues.
        groups = dict.fromkeys([*groups, *(SITUATION + (field,) for field in fields)])
        self.tables = {
            keys: train.groupby(list(keys), observed=True)["target"].agg(["sum", "count"])
            for keys in groups
        }

    def predict(self, data):
        prefix = np.ones(len(data), dtype=bool)
        probability = np.full(len(data), self.mean)
        for index, field in enumerate((None, *self.fields)):
            groups = SITUATION + self.fields[:index]
            if field is None:
                self._update(data, groups, prefix, probability)
                continue
            present = data[field].notna().to_numpy()
            self._update(data, SITUATION + (field,), present & ~prefix, probability)
            prefix &= present
            self._update(data, groups, prefix, probability)
        return probability

    def _update(self, data, keys, mask, probability):
        if mask.any():
            idx = pd.MultiIndex.from_frame(data.loc[mask, list(keys)])
            hits, count = self.tables[keys].reindex(idx, fill_value=0).to_numpy().T
            probability[mask] = (hits + self.strength * probability[mask]) / (count + self.strength)


def scores(actual, probability):
    n = len(actual)
    if not n:
        return {"n": 0}
    correct = int(np.sum((probability >= 0.5) == actual))
    rate = correct / n
    z = 1.96
    lower = (rate + z**2 / (2 * n) - z * sqrt(rate * (1 - rate) / n + z**2 / (4 * n**2))) / (
        1 + z**2 / n
    )
    return {
        "n": n,
        "correct": correct,
        "accuracy": rate,
        "log_loss": float(-np.mean(np.where(actual, np.log(probability), np.log1p(-probability)))),
        "wilson_lower_95": max(0.0, lower),
    }


def covered(data, probability, threshold, fields):
    mask = (np.maximum(probability, 1 - probability) >= threshold) & data[list(fields)].notna().all(
        axis=1
    )
    actual = data.loc[mask, "target"].to_numpy()
    correct = (probability[mask] >= 0.5) == actual
    by_game = (
        data.loc[mask, ["game_id"]]
        .assign(correct=correct)
        .groupby("game_id")["correct"]
        .agg(["sum", "count"])
    )
    lower = None
    if len(by_game) >= 30:
        rng = np.random.default_rng(85029)
        draws = rng.integers(0, len(by_game), size=(2000, len(by_game)))
        wins, counts = by_game.to_numpy()[draws].sum(axis=1).T
        lower = float(np.quantile(wins / counts, 0.025))
    return {
        "threshold": threshold,
        "coverage": float(mask.mean()),
        "games": len(by_game),
        "game_bootstrap_lower_95": lower,
        **scores(actual, probability[mask]),
    }


def evaluate(plays, cutoff=CUTOFF):
    dates = pd.to_datetime(plays["game_date"])
    usable = plays.loc[dates.le(pd.Timestamp(cutoff)) & plays["opening_drive"]].copy()
    data = features(usable.assign(target=usable["call"].eq("pass")))
    train = data.loc[data["season"].lt(2024)]
    validation = data.loc[data["season"].eq(2024)]
    test = data.loc[data["season"].gt(2024)]
    for earlier, later in ((train, validation), (validation, test)):
        if (
            earlier.empty
            or later.empty
            or pd.to_datetime(earlier["game_date"]).max()
            >= pd.to_datetime(later["game_date"]).min()
            or set(earlier["game_id"]).intersection(later["game_id"])
        ):
            raise ValueError("Pre-snap evaluation needs nonempty, chronological game splits.")

    fitted = {}
    for name, fields in CANDIDATES.items():
        options = [ObservedModel(train, fields, strength) for strength in (12, 32, 80)]
        fitted[name] = min(
            options, key=lambda m: scores(validation["target"], m.predict(validation))["log_loss"]
        )
    validation_scores = {
        name: scores(validation["target"], model.predict(validation))
        for name, model in fitted.items()
    }
    selected = min(validation_scores, key=lambda name: validation_scores[name]["log_loss"])
    model = fitted[selected]
    thresholds = (0.70, 0.75, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97)
    val_probabilities = model.predict(validation)
    val_coverage = [
        covered(validation, val_probabilities, level, model.fields) for level in thresholds
    ]
    validated = next(
        (row for row in val_coverage if (row["game_bootstrap_lower_95"] or 0) >= 0.90), None
    )
    probabilities = model.predict(test)
    first = ~test.duplicated(["game_id", "posteam"])
    atlanta = test["posteam"].eq("ATL")
    minnesota = test["defteam"].eq("MIN")
    subsets = {
        "all": np.ones(len(test), dtype=bool),
        "2025": test["season"].eq(2025),
        "2026": test["season"].eq(2026),
        "Atlanta": atlanta,
        "Atlanta first eligible snap": atlanta & first,
        "Minnesota defense": minnesota,
    }
    baseline = fitted["Situation only"].predict(test)
    return model, {
        "cutoff": cutoff,
        "selected": selected,
        "fields": model.fields,
        "strength": model.strength,
        "split": {
            name: {"plays": len(part), "games": int(part.game_id.nunique())}
            for name, part in (("train", train), ("validation", validation), ("test", test))
        },
        "validation": validation_scores,
        "test": {
            name: {
                "situation": scores(test.loc[mask, "target"], baseline[mask]),
                "observed": scores(test.loc[mask, "target"], probabilities[mask]),
            }
            for name, mask in subsets.items()
        },
        "test_label_counts_by_season": test.groupby("season")[list(OBSERVABLES)].count().to_dict(),
        "threshold_validated_for_90_percent": validated,
        "test_at_validated_threshold": (
            covered(test, probabilities, validated["threshold"], model.fields)
            if validated
            else None
        ),
        "test_at_validated_threshold_atlanta": (
            covered(test.loc[atlanta], probabilities[atlanta], validated["threshold"], model.fields)
            if validated
            else None
        ),
        "test_confidence_bands": [
            covered(test, probabilities, threshold, model.fields) for threshold in thresholds
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/presnap-results.json"))
    args = parser.parse_args()
    store = DataStore()
    _, result = evaluate(store.plays())
    result["dataset_version"] = store.manifest()["dataset_version"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

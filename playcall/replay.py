"""Replay held-out Atlanta opening drives without updating the frozen tree."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from playcall.data import DataStore, _atomic_json
from playcall.models import (
    SEED,
    PlaycallModels,
    RateModels,
    _git_commit,
    _ml_pipeline,
    _prepare,
    load_context,
)
from playcall.simulation import GameState, opening_tree


def _optional_int(value):
    return int(value) if pd.notna(value) else None


def _state(row):
    return GameState(
        down=int(row["down"]),
        distance=int(row["ydstogo"]),
        yardline_100=int(row["yardline_100"]),
        score_differential=int(row["score_differential"]),
        game_seconds_remaining=int(row["game_seconds_remaining"]),
        offense=row["posteam"],
        defense=row["defteam"],
        previous_call=None if row["previous_call"] == "none" else row["previous_call"],
        offense_rest=_optional_int(row["offense_rest"]),
        defense_rest=_optional_int(row["defense_rest"]),
        is_home=bool(row["is_home"]),
        caller=row["caller"],
    )


def _turnover(play):
    return any(
        pd.notna(play.get(key)) and play[key] == 1 for key in ("interception", "fumble_lost")
    )


def _same_outcome(branch, play):
    return (
        pd.notna(play["yards_gained"])
        and branch["call"] == play["call"]
        and branch["outcome"]["yards"] == int(play["yards_gained"])
        and branch["outcome"]["turnover"] == _turnover(play)
    )


def _frozen_evidence(tree, first, second=None):
    primary = tree["root"]
    surviving = [
        branch for branch in primary["children"] if branch["node"]["prediction"] is not None
    ]
    survival_mass = sum(branch["node"]["mass"] for branch in surviving)
    second_probability = None
    if survival_mass:
        second_probability = (
            sum(
                branch["node"]["mass"] * branch["node"]["prediction"]["p_pass"]
                for branch in surviving
            )
            / survival_mass
        )
    first_match = [branch for branch in primary["children"] if _same_outcome(branch, first)]
    two_calls = two_outcomes = False
    if second is not None:
        two_calls = any(
            child["call"] == second["call"]
            for branch in primary["children"]
            if branch["call"] == first["call"]
            for child in branch["node"]["children"]
        )
        two_outcomes = any(
            _same_outcome(child, second)
            for branch in first_match
            for child in branch["node"]["children"]
        )
    return {
        "frozen_first_p_pass": primary["prediction"]["p_pass"],
        "frozen_second_p_pass": second_probability,
        "frozen_second_conditioning_mass": survival_mass,
        "total_beam_mass": tree["retained_mass"],
        "first_call_and_outcome_in_beam": bool(first_match),
        "two_calls_in_beam": two_calls,
        "two_calls_and_outcomes_in_beam": two_outcomes,
    }


def _score(rows, label, prediction):
    if rows.empty:
        return {"n": 0, "accuracy": None, "log_loss": None, "brier": None}
    actual = rows[label].astype(int)
    probability = rows[prediction]
    return {
        "n": len(rows),
        "accuracy": float(accuracy_score(actual, probability.ge(0.5))),
        "log_loss": float(log_loss(actual, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(actual, probability)),
    }


def replay(store, as_of=None):
    context = load_context()
    as_of = pd.Timestamp(as_of or context["as_of"]).date().isoformat()
    frame = _prepare(store, as_of, context)
    train = frame.loc[frame["season"].between(2022, 2024)]
    historical = train.loc[train["opening_drive"]]
    test = frame.loc[frame["season"].between(2025, 2026) & frame["opening_drive"]]
    test = test.loc[test["posteam"].eq(context["game"]["away"])]
    if set(train["season"].unique()) != {2022, 2023, 2024} or test.empty:
        raise ValueError(
            "Refresh 2022–2026 nflverse data before replaying the held-out Atlanta drives."
        )
    availability = pd.Timestamp("2025-09-04", tz="UTC")
    rates = RateModels(train, historical, context, availability)
    estimator = _ml_pipeline("xgboost").fit(historical, historical["target"])
    dataset = store.manifest()["dataset_version"]
    model = PlaycallModels(
        rates,
        {"xgboost": estimator},
        historical,
        None,
        {
            "selected_model": "xgboost",
            "ensemble_analog_weight": 0.0,
            "model_version": "replay-2022-24",
            "dataset_version": dataset,
        },
        context,
    )
    league_order = historical.groupby(["game_id", "posteam"]).cumcount()
    first_reference = float(historical.loc[league_order.eq(0), "target"].mean())
    second_reference = float(historical.loc[league_order.eq(1), "target"].mean())
    rows = []
    for _, drive in test.groupby(["game_id", "posteam"], sort=False):
        first = drive.iloc[0]
        second = drive.iloc[1] if len(drive) > 1 else None
        tree = opening_tree(model, _state(first), depth=2, beam_width=8)
        row = _frozen_evidence(tree, first, second)
        row.update(
            game_id=first["game_id"],
            game_date=str(first["game_date"]),
            actual_first=int(first["target"]),
            actual_second=None if second is None else int(second["target"]),
            first_yards=_optional_int(first["yards_gained"]),
            second_yards=None if second is None else _optional_int(second["yards_gained"]),
            first_turnover=_turnover(first),
            eligible_drive_plays=len(drive),
            league_first_p_pass=first_reference,
            league_second_p_pass=second_reference,
            observed_state_second_p_pass=None
            if second is None
            else model.call_probabilities(_state(second))["p_pass"],
        )
        rows.append(row)
    results = pd.DataFrame(rows)
    actual_seconds = results.loc[results["actual_second"].notna()]
    paired = actual_seconds.loc[actual_seconds["frozen_second_p_pass"].notna()]
    return {
        "dataset_version": dataset,
        "git_head_at_run": _git_commit(),
        "replay_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "seed": SEED,
        "as_of": as_of,
        "configuration": {
            "train_seasons": [2022, 2023, 2024],
            "test_seasons": [2025, 2026],
            "model": "XGBoost with the project classifier hyperparameters, refit through 2024",
            "calibration": "Identity. This replay estimates performance without a held-out calibration stage.",
            "depth": 2,
            "beam_width": 8,
            "train_opening_calls": len(historical),
        },
        "first_call": {
            "xgboost": _score(results, "actual_first", "frozen_first_p_pass"),
            "league": _score(results, "actual_first", "league_first_p_pass"),
        },
        "second_call_all_actual": {
            "observed_state": _score(
                actual_seconds, "actual_second", "observed_state_second_p_pass"
            ),
            "league": _score(actual_seconds, "actual_second", "league_second_p_pass"),
        },
        "second_call_paired": {
            "frozen": _score(paired, "actual_second", "frozen_second_p_pass"),
            "observed_state": _score(paired, "actual_second", "observed_state_second_p_pass"),
            "league": _score(paired, "actual_second", "league_second_p_pass"),
        },
        "coverage": {
            "opening_drives": len(results),
            "drives_with_two_eligible_calls": len(actual_seconds),
            "one_eligible_call_drives": int(results["eligible_drive_plays"].eq(1).sum()),
            "first_call_turnovers": int(results["first_turnover"].sum()),
            "retained_first_call_and_exact_outcome": int(
                results["first_call_and_outcome_in_beam"].sum()
            ),
            "retained_two_call_sequence": int(actual_seconds["two_calls_in_beam"].sum()),
            "retained_two_call_sequence_and_exact_outcomes": int(
                actual_seconds["two_calls_and_outcomes_in_beam"].sum()
            ),
            "mean_depth2_retained_mass": float(results["total_beam_mass"].mean()),
            "mean_frozen_second_conditioning_mass_paired": (
                float(paired["frozen_second_conditioning_mass"].mean())
                if not paired.empty
                else None
            ),
        },
        "notes": [
            "The first eligible scrimmage state initializes the tree. No observed gains or second-play context enter it.",
            "Frozen second-call probabilities average the first-layer nonterminal branch predictions using retained mass.",
            "The observed-state comparator uses the actual state immediately before the second eligible call.",
            "Special teams, no-plays, and plays without a run/pass label are outside scope. A drive with one eligible call "
            "is excluded from second-call scores. Outcome coverage matches call, exact gain, and turnover status.",
            "The league reference uses separate pre-2025 marginal pass rates at the first and second eligible calls.",
        ],
        "drives": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data", help="Cached nflverse data directory")
    parser.add_argument("--as-of", help="Last included game date, inclusive")
    parser.add_argument("--output", type=Path, default=Path("docs/replay-results.json"))
    args = parser.parse_args()
    report = replay(DataStore(args.data), args.as_of)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, report)
    print(
        json.dumps(
            {key: report[key] for key in ("first_call", "second_call_paired", "coverage")}, indent=2
        )
    )


if __name__ == "__main__":
    main()

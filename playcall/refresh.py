"""Download public NFL releases and selectively rebuild the matchup model."""

import argparse
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filelock import FileLock

from playcall.data import DataStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", default=list(range(2022, 2027)))
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--freeze-updates",
        action="store_true",
        help="Save a new pregame forecast after retraining.",
    )
    args = parser.parse_args()
    if args.freeze_updates and not args.train:
        parser.error("--freeze-updates requires --train")
    Path("data").mkdir(exist_ok=True)
    with FileLock("data/refresh.lock", timeout=120):
        refresh(args)


def refresh(args):
    store = DataStore()
    results = store.refresh(args.seasons, force=args.force_download)
    print({key: value["status"] for key, value in results.items()})
    if not args.train:
        return
    from playcall import models

    context_path = Path(__file__).with_name("context.json")
    kickoff = datetime.fromisoformat(json.loads(context_path.read_text())["game"]["kickoff_utc"])
    today = datetime.now(timezone.utc).date()

    stamp = Path("models/refresh.json")
    manifest = store.manifest()
    # Only inputs used by the fitted model should trigger the expensive step.
    inputs = {
        key: value.get("sha256")
        for key, value in manifest["sources"].items()
        if key == "schedules" or key.startswith(("pbp_", "participation_", "charting_"))
    }
    inputs["context"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
    for name in ("models", "data"):
        inputs[name] = hashlib.sha256(
            Path(__file__).with_name(name + ".py").read_bytes()
        ).hexdigest()
    previous = json.loads(stamp.read_text()) if stamp.exists() else {}
    if previous.get("inputs") == inputs and all(
        Path(path).exists() for path in ("models/playcall.joblib", "models/backtest.json")
    ):
        print("Training inputs match the fitted model.")
        return
    # Exclude game-day observations even if this command is run after kickoff.

    cutoff = min(today, kickoff.date() - timedelta(days=1)).isoformat()
    report = models.fit_models(store, cutoff=cutoff)
    if args.freeze_updates and datetime.now(timezone.utc) < kickoff:
        from playcall.simulation import GameState, freeze_forecast

        game = models.load_context()["game"]
        schedule = store.schedules()
        match = schedule.loc[schedule["game_id"].eq(game["game_id"])]
        rest = {}
        if not match.empty:
            import pandas as pd

            for source, key in [("away_rest", "offense_rest"), ("home_rest", "defense_rest")]:
                value = match.iloc[0].get(source)
                if pd.notna(value):
                    rest[key] = int(value)
        snapshot = freeze_forecast(
            models.load_models(), GameState(**rest), game["game_id"], kickoff
        )
        print("Saved pregame scenario:", snapshot["id"])
    stamp.parent.mkdir(exist_ok=True)
    tmp = stamp.with_suffix(".tmp")
    tmp.write_text(
        json.dumps(
            {"inputs": inputs, "cutoff": cutoff, "dataset_version": manifest["dataset_version"]},
            indent=2,
        )
    )
    tmp.replace(stamp)
    print("Model rebuilt for cutoff", cutoff, ". Report fields:", list(report))


if __name__ == "__main__":
    main()

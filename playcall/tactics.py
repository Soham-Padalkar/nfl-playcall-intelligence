"""Estimate coherent personnel, formation and broad play-family combinations."""

from dataclasses import asdict, is_dataclass

import numpy as np
import pandas as pd

from playcall.presnap import personnel_code

PROFILE = ["personnel", "formation", "concept"]


def _label(frame, column):
    return (
        frame.get(column, pd.Series(pd.NA, index=frame.index))
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )


def _flag(frame, column):
    return _label(frame, column).str.lower().isin(["true", "1", "1.0"])


def charted_setups(plays):
    """Attach labels to the same snap. Missing charting stays missing."""
    personnel = _label(plays, "offense_personnel").map(lambda value: personnel_code(value, True))
    formation = _label(plays, "offense_formation").str.replace("_", " ").str.title()
    alignment = _label(plays, "qb_location").map(
        {"S": "Shotgun", "U": "Under Center", "P": "Pistol"}
    )
    detailed = formation.notna() & alignment.notna() & formation.ne(alignment).fillna(False)
    formation = formation.where(~detailed, formation + " · " + alignment).fillna(alignment)
    direction = _label(plays, "run_location").str.title()
    gap = _label(plays, "run_gap").str.title()
    run = (direction + " " + gap + " run").fillna(direction + " run")
    level = _label(plays, "pass_length").str.title()
    target = _label(plays, "pass_location").str.lower()
    passing = (level + " pass " + target).fillna(level + " pass")
    route = _label(plays, "route").str.title()
    passing = passing.where(route.isna(), route + " target")
    passing = passing.mask(_flag(plays, "is_play_action"), "Play-action pass")
    passing = passing.mask(_flag(plays, "is_screen_pass"), "Screen pass")
    passing = passing.mask(_flag(plays, "qb_scramble") | _flag(plays, "sack"))
    concept = run.where(plays["call"].eq("run"), passing)
    concept = concept.mask(
        _flag(plays, "is_rpo"), plays["call"].map({"run": "RPO run", "pass": "RPO pass"})
    )
    return plays.assign(personnel=personnel.astype("string"), formation=formation, concept=concept)


def setup_forecast(reference, state, prediction):
    """Rank historical combinations conditional on the upcoming run or pass."""
    item = asdict(state) if is_dataclass(state) else state
    same_offense = reference["posteam"].eq(item.get("offense", "ATL"))
    same_caller = reference["caller"].eq(item.get("caller") or "Tommy Rees")
    down = item.get("down", 1)
    distance = item.get("distance", 10)
    position = item.get("yardline_100", 75)
    previous = item.get("previous_call")
    weights = np.where(reference["down"].eq(down), 1, 0.12)
    weights *= np.exp(-reference["ydstogo"].sub(distance).abs().to_numpy() / 7)
    weights *= np.exp(-reference["yardline_100"].sub(position).abs().to_numpy() / 40)
    weights *= np.where(same_offense, 6, 1) * np.where(same_caller, 3, 1)
    weights *= np.where(reference["defteam"].eq(item.get("defense", "MIN")), 1.5, 1)
    historical_previous = reference["previous_call"].fillna("none")
    weights *= np.where(historical_previous.eq(previous or "none"), 1.8, 1)
    weights *= np.power(0.8, reference["season"].max() - reference["season"].to_numpy())
    available = reference.assign(weight=weights)
    result = {}
    for call in ("run", "pass"):
        opportunities = available.loc[available["call"].eq(call)]
        known = opportunities.dropna(subset=PROFILE)
        if known.empty:
            continue
        total = float(opportunities["weight"].sum())
        labeled = float(known["weight"].sum())
        if not labeled:
            continue
        groups = (
            known.groupby(PROFILE, as_index=False, observed=True)
            .agg(weight=("weight", "sum"), samples=("weight", "size"))
            .sort_values("weight", ascending=False)
            .head(3)
        )
        result[call] = {
            "n": len(known),
            "atlanta_n": int(known["posteam"].eq("ATL").sum()),
            "label_coverage": labeled / total,
            "options": [
                {
                    **{field: getattr(row, field) for field in PROFILE},
                    "samples": int(row.samples),
                    "conditional_share": float(row.weight / labeled),
                    "weighted_share": float(prediction[f"p_{call}"] * row.weight / labeled),
                }
                for row in groups.itertuples(index=False)
            ],
        }
    return result

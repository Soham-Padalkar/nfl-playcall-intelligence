"""Conditional opening-drive trees and immutable local forecast records."""

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from uuid import uuid4

from playcall.models import _git_commit

PREDICTIONS_ROOT = Path("data/predictions")


@dataclass(frozen=True)
class GameState:
    down: int = 1
    distance: int = 10
    yardline_100: int = 75
    score_differential: int = 0
    game_seconds_remaining: int = 3600
    offense: str = "ATL"
    defense: str = "MIN"
    previous_call: str | None = None
    personnel: str | None = None
    offense_rest: int | None = None
    defense_rest: int | None = None
    is_home: bool = False
    drive_ended: bool = False
    end_reason: str | None = None
    caller: str | None = None
    recent_calls: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "recent_calls", tuple(self.recent_calls))
        if self.down not in range(1, 5) or self.distance not in range(1, 100):
            raise ValueError("Down must be 1–4 and distance must be 1–99 yards.")
        if self.yardline_100 not in range(1, 100):
            raise ValueError("Field position must be 1–99 yards from the opponent's goal line.")
        if self.game_seconds_remaining not in range(3601):
            raise ValueError("The game clock must be 0–3,600 seconds.")
        if self.offense == self.defense:
            raise ValueError("Offense and defense must be different teams.")


def advance_state(
    state, call, yards, *, turnover=False, touchdown=False, safety=False, elapsed_seconds=30
):
    """Apply standard scrimmage transitions and stop on known possession changes."""
    if state.drive_ended:
        raise ValueError("The drive has ended.")
    if call not in {"run", "pass", "punt", "field_goal"} or not -99 <= yards <= 99:
        raise ValueError("Enter a valid play type and gain between −99 and 99 yards.")
    if not 0 <= elapsed_seconds <= state.game_seconds_remaining:
        raise ValueError("Elapsed seconds must fit within the current game clock.")
    field = state.yardline_100 - yards
    clock = state.game_seconds_remaining - elapsed_seconds
    score = state.score_differential
    reason = None
    if call in {"punt", "field_goal"}:
        reason = call
    elif turnover:
        reason = "turnover"
    elif touchdown or field <= 0:
        reason, score = "touchdown", score + 6
    elif safety or field >= 100:
        reason, score = "safety", score - 2
    elif clock == 0 or (state.game_seconds_remaining > 1800 >= clock):
        reason = "end of half"
    down, distance = state.down, state.distance
    if reason is None:
        if yards >= min(state.yardline_100, state.distance):
            down, distance = 1, min(10, field)
        elif state.down == 4:
            reason = "turnover on downs"
        else:
            down, distance = state.down + 1, min(99, max(1, state.distance - yards))
    recent = state.recent_calls or (
        (state.previous_call,) if state.previous_call in {"run", "pass"} else ()
    )
    return replace(
        state,
        down=down,
        distance=distance,
        yardline_100=min(99, max(1, field)),
        score_differential=score,
        game_seconds_remaining=clock,
        previous_call=call,
        recent_calls=(*recent, call)[-3:] if call in {"run", "pass"} else recent,
        drive_ended=reason is not None,
        end_reason=reason,
    )


def _node(model, state, mass, detailed=False):
    prediction = None
    if not state.drive_ended and state.game_seconds_remaining:
        prediction = model.predict(state) if detailed else model.call_probabilities(state)
    return {"state": asdict(state), "prediction": prediction, "mass": mass, "children": []}


def opening_tree(model, state, depth=3, beam_width=8):
    """Grow a bounded beam of scrimmage branches without redistributing pruned mass."""
    if not 1 <= depth <= 6 or not 1 <= beam_width <= 64:
        raise ValueError("The tree supports depth 1–6 and beam width 1–64.")
    root = _node(model, state, 1.0, detailed=True)
    frontier = [(state, root)]
    pruned_mass = 0.0
    for _ in range(depth):
        candidates = []
        for previous, node in frontier:
            prediction = node["prediction"]
            if prediction is None:
                continue
            for call in ("run", "pass"):
                for outcome in model.outcome_distribution(previous, call):
                    conditional = prediction[f"p_{call}"] * outcome["probability"]
                    if conditional > 0:
                        candidates.append(
                            (conditional * node["mass"], conditional, call, outcome, previous, node)
                        )
        ranked = sorted(candidates, key=lambda item: item[0], reverse=True)
        pruned_mass += sum(item[0] for item in ranked[beam_width:])
        frontier = []
        for mass, probability, call, outcome, previous, node in ranked[:beam_width]:
            following = advance_state(
                previous,
                call,
                outcome["yards"],
                turnover=outcome["turnover"],
                touchdown=outcome["touchdown"],
                safety=outcome["safety"],
                elapsed_seconds=min(30, previous.game_seconds_remaining),
            )
            next_node = _node(model, following, mass)
            node["children"].append(
                {
                    "call": call,
                    "outcome": outcome,
                    "conditional_probability": probability,
                    "node": next_node,
                }
            )
            frontier.append((following, next_node))
    return {
        "root": root,
        "depth": depth,
        "beam_width": beam_width,
        "retained_mass": max(0.0, 1.0 - pruned_mass),
        "pruned_mass": min(1.0, pruned_mass),
        "scope": "Run or pass conditional on a scrimmage decision; simulated plays use 30 seconds.",
    }


def _save_record(payload, root):
    root = Path(root)
    path = root / payload["kind"]
    path.mkdir(parents=True, exist_ok=True)
    destination = path / f"{payload['id']}.json"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(payload, stream, indent=2, allow_nan=False)
        except (TypeError, ValueError):
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        # A same-filesystem hard link publishes a complete record without replacing an existing one.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def _record(model, game_id, kind, now):
    if not game_id or not game_id.strip():
        raise ValueError("A game identifier is required.")
    return {
        "id": uuid4().hex,
        "kind": kind,
        "game_id": game_id,
        "created_at": now.isoformat(),
        "model_version": model.version,
        "dataset_version": model.dataset_version,
        "git_commit": _git_commit(),
        "training_git_commit": model.report["git_commit"],
        "schema_version": 1,
    }


def freeze_forecast(model, state, game_id, kickoff_utc, root=PREDICTIONS_ROOT, depth=3):
    now = datetime.now(UTC)
    kickoff = datetime.fromisoformat(kickoff_utc) if isinstance(kickoff_utc, str) else kickoff_utc
    if kickoff.tzinfo is None or now >= kickoff:
        raise ValueError("Pre-game forecasts require a timezone-aware kickoff in the future.")
    forecast = opening_tree(model, state, depth=depth)
    now = datetime.now(UTC)
    if now >= kickoff:
        raise ValueError("Kickoff passed before this forecast could be frozen.")
    payload = _record(model, game_id, "pregame", now)
    payload.update(
        state=asdict(state), kickoff_utc=kickoff.astimezone(UTC).isoformat(), forecast=forecast
    )
    return _save_record(payload, root)


def list_forecasts(root=PREDICTIONS_ROOT, game_id=None, kind=None):
    if kind not in {None, "pregame", "adaptive"}:
        raise ValueError("Forecast type must be pregame or adaptive.")
    kinds = [kind] if kind else ["pregame", "adaptive"]
    records = []
    for category in kinds:
        for path in (Path(root) / category).glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if game_id is None or payload["game_id"] == game_id:
                records.append(payload)
    return sorted(records, key=lambda record: (record["created_at"], record["id"]))


def record_adaptive(
    model,
    state,
    actual_call,
    yards,
    game_id,
    root=PREDICTIONS_ROOT,
    *,
    turnover=False,
    touchdown=False,
    elapsed_seconds=30,
    parent_id=None,
    depth=3,
):
    state_before = asdict(state)
    prediction_before = model.predict(state)
    if parent_id:
        records = list_forecasts(root, game_id=game_id)
        parent = next((record for record in records if record["id"] == parent_id), None)
        if parent is None:
            raise ValueError("The parent forecast must exist for the same game.")
        if GameState(**parent["state"]) != state:
            raise ValueError("The starting state must match the parent forecast.")
        prediction_before = parent["forecast"]["root"]["prediction"]
    following = advance_state(
        state,
        actual_call,
        yards,
        turnover=turnover,
        touchdown=touchdown,
        elapsed_seconds=elapsed_seconds,
    )
    forecast = opening_tree(model, following, depth=depth)
    payload = _record(model, game_id, "adaptive", datetime.now(UTC))
    payload.update(
        parent_id=parent_id,
        state_before=state_before,
        state=asdict(following),
        forecast=forecast,
        prediction_before=prediction_before,
        actual={
            "call": actual_call,
            "yards": yards,
            "turnover": turnover,
            "touchdown": touchdown,
            "elapsed_seconds": elapsed_seconds,
        },
    )
    return _save_record(payload, root)

"""Pregame forecasts, decision-tree exploration, and adaptive evaluation."""

import json
from datetime import UTC, datetime
from math import log

import pandas as pd
import streamlit as st

from playcall import ui

PREDICTIONS = ui.DATA / "predictions"


def _state_inputs(include_depth=True):
    from playcall.simulation import GameState

    cols = st.columns(3)
    down = cols[0].selectbox("Down", [1, 2, 3, 4])
    distance = cols[1].number_input("Yards to gain", 1, 99, 10)
    position = cols[2].number_input("Yards to Minnesota’s goal line", 1, 99, 75)
    cols = st.columns(3)
    seconds = cols[0].number_input("Game clock (seconds remaining)", 1, 3600, 3600)
    score = cols[1].number_input("Atlanta score differential", -99, 99, 0)
    previous = cols[2].selectbox(
        "Previous play",
        [None, "run", "pass"],
        format_func=lambda value: value.title() if value else "None",
    )
    personnel = st.text_input(
        "Offensive personnel", placeholder="Use the charted label, if available"
    )
    with st.expander("Rest days and tree size" if include_depth else "Rest days"):
        cols = st.columns(3 if include_depth else 2)
        game_id = ui.context()["game"]["game_id"]
        away_rest, home_rest = ui.scheduled_rest(game_id)
        offense_rest = cols[0].number_input("Atlanta days of rest", 0, 24, value=away_rest)
        defense_rest = cols[1].number_input("Minnesota days of rest", 0, 24, value=home_rest)
        depth = cols[2].slider("Plays to forecast", 1, 5, 3) if include_depth else 3
    state = GameState(
        down=down,
        distance=min(distance, position),
        yardline_100=position,
        score_differential=score,
        game_seconds_remaining=seconds,
        offense="ATL",
        defense="MIN",
        previous_call=previous,
        personnel=personnel.strip() or None,
        offense_rest=offense_rest,
        defense_rest=defense_rest,
        is_home=False,
    )
    if distance > position:
        st.caption("Distance is set to goal-to-go for the forecast.")
    return state, depth


def _tree_node(node, path="opening"):
    children = node["children"]
    if not children:
        st.caption(
            f"End of drive · {node['state'].get('end_reason')}"
            if node["state"].get("drive_ended")
            else "Forecast depth reached or later branches pruned."
        )
        return
    options = []
    for child in children:
        outcome = child["outcome"]
        after = child["node"]["state"]
        result = after.get("end_reason") or f"down {after['down']}, {after['distance']} to go"
        options.append(
            f"{child['call'].title()} · {outcome.get('yards', '—')} yd · {result} · "
            f"{child['conditional_probability']:.1%} conditional"
        )
    chosen = st.selectbox(
        "Show a branch", list(range(len(children))), format_func=options.__getitem__, key=path
    )
    edge = children[chosen]
    active = edge["node"]
    state = active["state"]
    st.caption(
        f"Path probability {active['mass']:.1%} · {state['yardline_100']} yd to goal · "
        f"{state['game_seconds_remaining']} sec left"
    )
    if active.get("prediction"):
        ui.show_prediction(active["prediction"])
    _tree_node(active, f"{path}-{chosen}")


def _tree(tree, path="opening"):
    cols = st.columns(2)
    cols[0].metric("Represented probability mass", f"{tree['retained_mass']:.1%}")
    cols[1].metric("Pruned probability mass", f"{tree['pruned_mass']:.1%}")
    st.caption(tree["scope"])
    with st.expander("Explore conditional branches", expanded=True):
        children = tree["root"]["children"]
        if children:
            st.dataframe(
                [
                    {
                        "Opening call": edge["call"].title(),
                        "Yards": edge["outcome"]["yards"],
                        "Next down": edge["node"]["state"]["down"],
                        "Distance": edge["node"]["state"]["distance"],
                        "Drive result": edge["node"]["state"]["end_reason"],
                        "Path probability": edge["conditional_probability"],
                    }
                    for edge in children
                ],
                width="stretch",
                hide_index=True,
                column_config={
                    "Path probability": st.column_config.NumberColumn(
                        "Path probability", format="percent"
                    )
                },
            )
        _tree_node(tree["root"], path)


def predictor():
    from playcall.simulation import freeze_forecast, opening_tree

    st.caption("Change the pre-snap state to see updated opening-drive probabilities.")
    state, depth = _state_inputs()
    with st.expander("Update from the visible setup"):
        _observed_prediction(state)
    model = ui.require_model()
    if model is None:
        return
    tree = opening_tree(model, state, depth=depth)
    st.subheader("Next eligible play")
    ui.show_prediction(tree["root"]["prediction"])
    st.subheader("Opening-drive branches")
    _tree(tree)
    game = ui.context()["game"]
    if datetime.now(UTC) >= datetime.fromisoformat(game["kickoff_utc"]):
        st.info(
            "Kickoff has passed. You can explore scenarios and review any pregame snapshots in Live Prediction & Evaluation."
        )
        return
    if st.button("Freeze pregame forecast", type="primary"):
        try:
            snapshot = freeze_forecast(
                model, state, game["game_id"], game["kickoff_utc"], root=PREDICTIONS, depth=depth
            )
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error))
        else:
            st.success(
                f"Saved pregame forecast {snapshot['id']}. Open Live Prediction & Evaluation to review it."
            )


def _observed_prediction(state):
    from playcall.presnap import OBSERVABLES, features

    st.caption("Enter only the setup you can see for this play. Leave uncertain fields blank.")
    report_path = ui.ROOT / "docs" / "presnap-results.json"
    if not report_path.is_file():
        st.info("The pre-snap evaluation report is missing.")
        return
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fields = tuple(report["fields"])
    with st.form("observed_setup"):
        columns = st.columns(3)
        inputs = {}
        options = {
            "qb_location": {"U": "Under center", "S": "Shotgun", "P": "Pistol"},
            "starting_hash": {"L": "Left", "M": "Middle", "R": "Right"},
            "is_motion": {"True": "Observed", "False": "No motion observed before the snap"},
        }
        for index, field in enumerate(fields):
            column = columns[index % len(columns)]
            if field in options:
                labels = options[field]
                inputs[field] = column.selectbox(
                    OBSERVABLES[field],
                    [None, *labels],
                    format_func=lambda value, names=labels: names.get(value, "Unknown"),
                )
            elif field in ("n_defense_box", "n_offense_backfield"):
                inputs[field] = column.number_input(OBSERVABLES[field], 0, 11, value=None)
            else:
                inputs[field] = column.text_input(OBSERVABLES[field])
        ready = st.form_submit_button("Update run/pass probabilities")
    if not ready:
        return
    if not any(value not in (None, "") for value in inputs.values()):
        st.info("Enter at least one observation.")
        return
    model = _load_observed_model(ui.get_manifest()["dataset_version"], fields, report["strength"])
    if model is None:
        st.info("Load the 2022 and 2023 play-by-play and charting data first.")
        return
    row = features(
        pd.DataFrame(
            [
                {
                    **inputs,
                    "down": state.down,
                    "ydstogo": state.distance,
                    "yardline_100": state.yardline_100,
                }
            ]
        )
    )
    p_pass = float(model.predict(row)[0])
    ui.show_prediction({"p_pass": p_pass, "p_run": 1 - p_pass})
    band = report["test_at_validated_threshold"]
    if (
        band
        and (band["game_bootstrap_lower_95"] or 0) >= 0.90
        and row[list(fields)].notna().all(axis=None)
    ):
        if max(p_pass, 1 - p_pass) >= band["threshold"]:
            st.caption(
                f"This estimate is in the high-confidence group. On held-out league games, "
                f"{band['accuracy']:.1%} of estimates in that group were correct across "
                f"{band['coverage']:.1%} of all eligible plays."
            )


@st.cache_resource(show_spinner="Fitting the pre-snap model…", max_entries=2)
def _load_observed_model(dataset_version, fields, strength):
    from playcall.data import DataStore
    from playcall.presnap import ObservedModel, features

    data = DataStore(ui.DATA).plays([2022, 2023])
    if data.empty:
        return None
    opening = data.loc[data["opening_drive"]]
    if opening.empty or set(opening["season"]) != {2022, 2023}:
        return None
    return ObservedModel(
        features(opening.assign(target=opening["call"].eq("pass"))), fields, strength
    )


def _state_line(state):
    minutes, seconds = divmod(state.game_seconds_remaining, 60)
    st.caption(
        f"Down {state.down} · {state.distance} yd to gain · {state.yardline_100} yd to goal · "
        f"{minutes}:{seconds:02d} remaining · ATL margin {state.score_differential:+d}"
    )


def _series(events, frozen):
    """Compare the same event only where a frozen branch reaches the observed state."""
    rows = []
    candidates = [frozen["forecast"]["root"]] if frozen else []
    fields = (
        "down",
        "distance",
        "yardline_100",
        "score_differential",
        "game_seconds_remaining",
        "previous_call",
    )
    for index, event in enumerate(events, start=1):
        observed = event["actual"]
        prediction = event.get("prediction_before")
        adaptive = prediction["p_pass"] if prediction and not (frozen and index == 1) else None
        matched = [
            node
            for node in candidates
            if node.get("prediction")
            and all(node["state"].get(key) == event["state_before"].get(key) for key in fields)
        ]
        mass = sum(node["mass"] for node in matched)
        locked = (
            sum(node["mass"] * node["prediction"]["p_pass"] for node in matched) / mass
            if mass
            else None
        )
        if observed["call"] in {"run", "pass"}:
            rows.append(
                {
                    "Play": index,
                    "Actual": observed["call"],
                    "Yards": observed["yards"],
                    "Frozen": locked,
                    "Adaptive": adaptive,
                }
            )
        candidates = [
            child["node"]
            for node in matched
            for child in node["children"]
            if child["call"] == observed["call"]
        ]
    return pd.DataFrame(rows)


def _scores(frame, label):
    observed = frame.loc[frame[label].notna()]
    if observed.empty:
        return None
    truth = observed["Actual"].eq("pass").astype(int)
    probability = observed[label].astype(float)
    selected = probability.where(truth.eq(1), 1 - probability).clip(1e-15, 1.0)
    return {
        "Forecast": label,
        "Count": len(observed),
        "Accuracy": (probability.ge(0.5) == truth).mean(),
        "Brier score": ((probability - truth) ** 2).mean(),
        "Log loss": -selected.map(log).mean(),
        "Mean predicted pass": probability.mean(),
        "Observed pass share": truth.mean(),
    }


def _evaluation(events, frozen):
    if not events:
        return
    st.subheader("Forecast vs observed plays")
    frame = _series(events, frozen)
    if frame.empty:
        st.info("The log has no eligible run or pass to score.")
        return
    share = {
        key: st.column_config.NumberColumn(key, format="percent") for key in ("Frozen", "Adaptive")
    }
    st.dataframe(frame, width="stretch", hide_index=True, column_config=share)
    st.caption(
        "Frozen comparisons require an exact state match on a retained branch. "
        "Adaptive scores use forecasts saved after the preceding actual play, or the starting forecast for a drive with no frozen record."
    )
    scores = [row for label in ("Frozen", "Adaptive") if (row := _scores(frame, label))]
    if not scores:
        return
    formats = {
        key: st.column_config.NumberColumn(key, format="percent")
        for key in ("Accuracy", "Mean predicted pass", "Observed pass share")
    }
    st.dataframe(scores, width="stretch", hide_index=True, column_config=formats)
    with st.expander("Reliability plot"):
        roles = [row["Forecast"] for row in scores]
        label = st.selectbox("Forecast to score", roles)
        matching = frame.loc[frame[label].notna()].copy()
        matching["truth"] = matching["Actual"].eq("pass").astype(int)
        matching["bin"] = pd.cut(
            matching[label], bins=[0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True
        )
        rows = matching.groupby("bin", observed=True).agg(
            forecast=(label, "mean"), observed=("truth", "mean"), n=("truth", "size")
        )
        ui.reliability(rows, label)
        st.caption(
            "These bins describe this drive. Use the chronological backtest for a larger calibration sample."
        )


def _chains(records, origin):
    """Return the saved paths for this origin, including concurrent entries."""
    paths = {origin["id"] if origin else None: []}
    internal = set()
    for item in records:
        parent = item.get("parent_id")
        if parent in paths:
            paths[item["id"]] = [*paths[parent], item]
            internal.add(parent)
    return {end: chain for end, chain in paths.items() if end not in internal}


def live():
    from playcall.simulation import GameState, list_forecasts, record_adaptive

    game = ui.context()["game"]
    rehearsal = datetime.now(UTC) < datetime.fromisoformat(game["kickoff_utc"])
    game_id = game["game_id"] + "-rehearsal" if rehearsal else game["game_id"]
    st.caption(
        "Record each eligible play after it happens. Select a pregame forecast to compare it with the adaptive sequence."
    )
    if rehearsal:
        st.info("Kickoff is in the future. Play entries use a separate rehearsal log.")
    try:
        snapshots = list_forecasts(root=PREDICTIONS, game_id=game["game_id"], kind="pregame")
        records = list_forecasts(root=PREDICTIONS, game_id=game_id, kind="adaptive")
    except (ValueError, RuntimeError, OSError) as error:
        st.error(str(error))
        return
    choices = {row["id"]: row for row in snapshots}
    choice = (
        st.selectbox(
            "Frozen forecast" if rehearsal else "Forecast origin",
            [None, *choices],
            format_func=lambda key: (
                choices[key]["created_at"] if key else "No frozen forecast selected"
            ),
            index=len(choices),
        )
        if choices
        else None
    )
    origin = choices.get(choice)
    records = sorted(records, key=lambda item: item["created_at"])
    paths = _chains(records, None if rehearsal else origin)
    end = next(reversed(paths))
    if len(paths) > 1:
        end = st.selectbox(
            "Recorded sequence",
            list(paths),
            index=len(paths) - 1,
            format_func=lambda key: f"{paths[key][-1]['created_at']} · {len(paths[key])} plays",
        )
    events = paths[end]
    if origin:
        with st.expander("Locked pregame prediction"):
            st.caption(f"Saved {origin['created_at']}")
            ui.show_prediction(origin["forecast"]["root"]["prediction"])
            st.download_button(
                "Download frozen forecast",
                json.dumps(origin, indent=2),
                file_name=f"{origin['id']}.json",
                mime="application/json",
            )
    _evaluation(events, None if rehearsal else origin)
    if events:
        st.download_button(
            "Download play log",
            json.dumps(events, indent=2),
            file_name=f"{game_id}-play-log.json",
            mime="application/json",
        )
    state_record = events[-1] if events else (None if rehearsal else origin)
    state = GameState(**state_record["state"]) if state_record else GameState()
    model = ui.get_model()
    if state_record is None and model is not None:
        with st.expander("Set starting state"):
            state, _ = _state_inputs(include_depth=False)
    st.subheader("Current drive state")
    _state_line(state)
    if state.drive_ended:
        st.info(f"Drive ended · {state.end_reason}")
        return
    pending = state_record["forecast"]["root"]["prediction"] if state_record else None
    if pending:
        ui.show_prediction(pending)
    if model is None:
        st.info(
            "Train a model from the cached data to record a play and generate the next adaptive forecast."
        )
        return
    if pending is None:
        ui.show_prediction(model.predict(state))
    with st.form("record-actual"):
        cols = st.columns(3)
        actual_call = cols[0].selectbox(
            "Actual playcall",
            ["run", "pass", "punt", "field_goal"],
            format_func=lambda value: value.replace("_", " ").title(),
        )
        yards = cols[1].number_input("Yards gained", -99, 99, 0)
        available = min(120, state.game_seconds_remaining)
        elapsed = cols[2].number_input("Seconds elapsed", 0, available, min(30, available))
        result = st.radio(
            "Play result", ["Play continues", "Turnover", "Touchdown"], horizontal=True
        )
        submitted = st.form_submit_button("Record play and update", type="primary")
    if submitted:
        try:
            record_adaptive(
                model,
                state,
                actual_call,
                yards,
                game_id,
                root=PREDICTIONS,
                turnover=result == "Turnover",
                touchdown=result == "Touchdown",
                elapsed_seconds=elapsed,
                parent_id=state_record["id"] if state_record else None,
            )
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error))
        else:
            st.rerun()

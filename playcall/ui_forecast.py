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
        depth = cols[2].slider("Plays to forecast", 1, 6, 4) if include_depth else 3
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


def _likely_path(node, remaining):
    """Choose a complete retained path. A pruned continuation is not an end of drive."""
    if not node["children"]:
        if remaining == 0 or node["state"]["drive_ended"]:
            return node["mass"], []
        return None
    paths = []
    for edge in node["children"]:
        following = _likely_path(edge["node"], remaining - 1)
        if following is not None:
            mass, remainder = following
            paths.append((mass, [(node, edge), *remainder]))
    return max(paths, key=lambda item: item[0], default=None)


def _drive_plan(tree):
    projection = _likely_path(tree["root"], tree["depth"])
    if not projection or not projection[1]:
        return
    mass, plays = projection
    rows = []
    for number, (node, edge) in enumerate(plays, start=1):
        state = node["state"]
        call = edge["call"]
        setup = node["prediction"].get("setups", {}).get(call, {}).get("options", [])
        best = setup[0] if setup else {}
        rows.append(
            {
                "Snap": number,
                "Down": state["down"],
                "To gain": state["distance"],
                "Call": call.title(),
                "Personnel": best.get("personnel", "Unavailable"),
                "Formation": best.get("formation", "Unavailable"),
                "Concept family": best.get("concept", "Unavailable"),
                "Simulated gain": edge["outcome"]["yards"],
            }
        )
    st.dataframe(rows, hide_index=True)
    st.caption(
        f"Estimated path probability {mass:.2%}. Later snaps depend on earlier simulated gains. "
        "The simulation covers eligible run/pass decisions. Kicks can be logged when they occur."
    )


def _show_live_page():
    st.session_state["dashboard_page"] = "Live Prediction & Evaluation"


def predictor():
    from playcall.simulation import freeze_forecast, opening_tree

    st.caption("Explore Atlanta’s opening drive or switch to the live play log.")
    st.button("Open live play log", on_click=_show_live_page)
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
    _drive_plan(tree)
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


def _observed_prediction(state, recorded_calls=None):
    from playcall.neural_selection import FIELDS, MODEL
    from playcall.presnap import HISTORY, OBSERVABLES, features, play_history

    st.caption("Enter only the setup you can see for this play. Leave uncertain fields blank.")
    checkpoint = ui.MODEL.with_name(MODEL.name)
    if not checkpoint.is_file():
        st.info("Train the neural predictor with python -m playcall.neural_selection.")
        return
    with st.form("observed_setup"):
        columns = st.columns(3)
        inputs = {}
        options = {
            "qb_location": {"U": "Under center", "S": "Shotgun", "P": "Pistol"},
            "starting_hash": {"L": "Left", "M": "Middle", "R": "Right"},
            "is_motion": {"True": "Observed", "False": "No motion observed before the snap"},
        }
        visible = [field for field in FIELDS if field not in HISTORY]
        for index, field in enumerate(visible):
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
            elif field == "offense_personnel":
                inputs[field] = column.text_input(OBSERVABLES[field], placeholder="11, 12, 21")
            elif field == "defense_personnel":
                inputs[field] = column.text_input(OBSERVABLES[field], placeholder="4-2-5")
            else:
                inputs[field] = column.text_input(OBSERVABLES[field])
        if recorded_calls is None:
            st.caption(
                "Optional. Enter previous calls by the same offense in this game, oldest first."
            )
            recent = [
                col.selectbox(f"Previous call {i}", [None, "run", "pass"])
                for i, col in enumerate(st.columns(3), start=1)
            ]
        else:
            recent = [call for call in recorded_calls if call in {"run", "pass"}][-3:]
            if recent:
                st.caption("Earlier calls from this game · " + ", ".join(recent))
        inputs.update(play_history(recent))
        ready = st.form_submit_button("Update run/pass probabilities")
    if not ready:
        return
    try:
        model = _load_observed_model(checkpoint, checkpoint.stat().st_mtime_ns)
    except (OSError, ValueError, KeyError) as error:
        st.error(f"Could not load the neural predictor. {error}")
        return
    row = features(
        pd.DataFrame(
            [
                {
                    **inputs,
                    "down": state.down,
                    "ydstogo": state.distance,
                    "yardline_100": state.yardline_100,
                    "game_seconds_remaining": state.game_seconds_remaining,
                    "score_differential": state.score_differential,
                }
            ]
        )
    )
    p_pass = float(model.predict(row)[0])
    ui.show_prediction({"p_pass": p_pass, "p_run": 1 - p_pass, "selected_model": model.name})
    context = [
        f"{OBSERVABLES[field]}: {row.iloc[0][field]}"
        for field in ("offense_personnel", "defense_personnel")
        if pd.notna(row.iloc[0][field])
    ]
    if context:
        st.caption("Personnel entered · " + " · ".join(context))
    for field in ("offense_personnel", "defense_personnel"):
        if inputs.get(field, "").strip() and pd.isna(row.iloc[0][field]):
            st.warning(f"Check the format for {OBSERVABLES[field].lower()}.")


@st.cache_resource(show_spinner="Loading the neural predictor…", max_entries=2)
def _load_observed_model(path, signature):
    from playcall.neural_selection import load_winner

    return load_winner(path)


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
    from playcall.simulation import GameState, list_forecasts, opening_tree, record_adaptive

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
    tree = state_record["forecast"] if state_record else None
    pending = tree["root"]["prediction"] if tree else None
    if model is not None and (not pending or "setups" not in pending):
        tree = opening_tree(model, state, depth=3)
        pending = tree["root"]["prediction"]
    if pending:
        ui.show_prediction(pending)
    if tree:
        with st.expander("Updated drive scenario"):
            _drive_plan(tree)
    with st.expander("Update from the visible setup"):
        _observed_prediction(state, [event["actual"]["call"] for event in events])
    if model is None:
        st.info(
            "Train a model from the cached data to record a play and generate the next adaptive forecast."
        )
        return
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
        st.caption("Optional · Enter the official next spot if a penalty or ruling changed it.")
        cols = st.columns(3)
        official = (
            cols[0].number_input("Official next down", 1, 4, value=None),
            cols[1].number_input("Official yards to gain", 1, 99, value=None),
            cols[2].number_input("Official yards to goal", 1, 99, value=None),
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
                official_next=official if any(value is not None for value in official) else None,
            )
        except (ValueError, RuntimeError, OSError) as error:
            st.error(str(error))
        else:
            st.rerun()

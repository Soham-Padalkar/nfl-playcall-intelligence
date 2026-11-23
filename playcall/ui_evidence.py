"""Matchup evidence, defensive comparisons, and historical tendencies."""

from datetime import UTC, datetime
from html import escape
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import streamlit as st

from playcall import ui
from playcall.data import DataStore

INJURY_FIELDS = {
    "week": "Week",
    "full_name": "Player",
    "position": "Position",
    "report_primary_injury": "Injury",
    "report_status": "Game status",
    "practice_status": "Practice status",
}


@st.cache_data(max_entries=4, show_spinner=False)
def _roster_records(root, dataset_version, season, week, cutoff):
    store = DataStore(root)
    teams = ["ATL", "MIN"]
    injuries = store.injuries([season])
    if "season_type" in injuries:
        injuries = injuries.loc[injuries["season_type"].eq("REG")]
    current = injuries.loc[injuries["week"].eq(week)] if not injuries.empty else injuries
    depth = store.depth_charts([season])
    schedule = store.schedules([season])
    if not depth.empty and {"dt", "team", "pos_rank"}.issubset(depth):
        stamps = pd.to_datetime(depth["dt"], utc=True, errors="coerce")
        depth = depth.loc[depth["team"].isin(teams) & stamps.le(cutoff)].assign(Source_time=stamps)
        depth = depth.loc[
            depth["Source_time"].eq(depth.groupby("team")["Source_time"].transform("max"))
        ]
        depth = depth.loc[depth["pos_rank"].eq(1)]
    else:
        depth = pd.DataFrame()
    if not injuries.empty and not schedule.empty:
        completed = schedule.loc[
            schedule["home_score"].notna()
            & schedule["away_score"].notna()
            & pd.to_datetime(schedule["gameday"]).le(cutoff.tz_localize(None).normalize())
        ]
        team_weeks = pd.concat(
            [
                completed[["week", f"{side}_team"]].rename(columns={f"{side}_team": "team"})
                for side in ("home", "away")
            ]
        ).drop_duplicates()
        injuries = injuries.loc[injuries["team"].isin(teams) & injuries["week"].lt(week)].merge(
            team_weeks, on=["team", "week"]
        )
        injuries = injuries.loc[
            injuries["week"].eq(injuries.groupby("team")["week"].transform("max"))
        ]
    else:
        injuries = pd.DataFrame()
    return depth, injuries, current


def _injury_table(injuries, team, empty_message):
    if not injuries.empty and team in injuries["team"].values:
        fields = {key: value for key, value in INJURY_FIELDS.items() if key in injuries}
        st.dataframe(
            injuries.loc[injuries["team"].eq(team), list(fields)].rename(columns=fields),
            width="stretch",
            hide_index=True,
        )
    else:
        st.info(empty_message)


def overview():
    context = ui.context()
    game = context["game"]
    kickoff = datetime.fromisoformat(game["kickoff_utc"]).astimezone(ZoneInfo("America/Chicago"))
    schedule = ui.target_schedule(game["game_id"])
    week = int(schedule.get("week", game["game_id"].split("_")[1]))
    event = f"{kickoff:%A, %B %d, %Y · %I:%M %p %Z} · {game['stadium']}"
    st.markdown(
        f'<div class="matchup-banner"><small>WEEK {week} · NFL REGULAR SEASON</small>'
        f"<strong>Atlanta Falcons at Minnesota Vikings</strong><span>{escape(event)}</span></div>",
        unsafe_allow_html=True,
    )
    manifest = ui.get_manifest()
    model = ui.get_model()
    cards = st.columns(3)
    cards[0].metric("Atlanta playcaller", context["caller"])
    cards[1].metric("Minnesota defensive coordinator", context["defensive_coordinator"])
    cards[2].metric("Prediction model", "Ready" if model else "Training required")
    last = max(
        (row["retrieved_at"] for row in manifest["sources"].values() if row.get("retrieved_at")),
        default=None,
    )
    if last:
        st.caption(f"Dataset {manifest['dataset_version']} · Last local retrieval {last}")
    st.markdown(
        f"[Falcons schedule]({context['sources']['schedule']}) · "
        f"[Flores coaching profile]({context['sources']['coordinator']})"
    )
    st.subheader("Opening possession")
    if model:
        from playcall.simulation import GameState

        away_rest, home_rest = ui.scheduled_rest(game["game_id"])
        ui.show_prediction(model.predict(GameState(offense_rest=away_rest, defense_rest=home_rest)))
        st.caption("Assumed starting state · 1st and 10 at Atlanta’s 25, tied score, 60:00 left")
        if away_rest is not None and home_rest is not None:
            st.caption(f"Scheduled rest · Atlanta {away_rest} days · Minnesota {home_rest} days")
    else:
        st.info("Train the local model to generate an opening-possession forecast.")
    st.subheader("Roster and injury context")
    st.caption(f"Reviewed {context['as_of']}")
    for note in context["notes"]:
        st.write(note)
    st.markdown(
        f"[Atlanta depth chart]({context['sources']['depth_chart']}) · "
        f"[Minnesota injury report]({context['sources']['injuries']})"
    )
    with st.expander("Latest cached lineup and injury records"):
        before = min(pd.Timestamp.now(tz=UTC), pd.Timestamp(game["kickoff_utc"]))
        depth, injuries, current = _roster_records(
            ui.DATA, manifest["dataset_version"], kickoff.year, week, before.floor("h")
        )
        team = st.selectbox("Team records", [game["away"], game["home"]])
        st.caption(
            "Depth ranks are roster projections. Injury reports show reported status by week. "
            "The final lineup depends on game-day availability."
        )
        st.markdown("**First on depth chart**")
        if not depth.empty and team in depth["team"].values:
            fields = {
                "Source_time": "Source time (UTC)",
                "pos_grp": "Unit",
                "pos_name": "Position",
                "player_name": "Player",
            }
            st.dataframe(
                depth.loc[depth["team"].eq(team), list(fields)].rename(columns=fields),
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No dated depth chart is cached for this team.")
        st.markdown(f"**Week {week} injury reports cached before kickoff**")
        source = manifest["sources"].get(f"injuries_{kickoff.year}", {})
        retrieved = pd.to_datetime(source.get("retrieved_at"), utc=True, errors="coerce")
        if pd.notna(retrieved) and retrieved <= before and retrieved < kickoff:
            _injury_table(current, team, f"No Week {week} injury report is cached for this team.")
            st.caption(
                f"nflverse injuries · Local retrieval {retrieved:%Y-%m-%d %H:%M UTC}. "
                f"The source supplies a week and status. [Injury data file]({source['url']})"
            )
        else:
            st.info("No injury file with a verified pregame retrieval time is available.")
        st.markdown("**Injury reports from the latest earlier completed week**")
        _injury_table(injuries, team, "No completed-week injury report is cached for this team.")
    with st.expander("Data coverage and retrieval dates"):
        ui.source_table(manifest)


def caller_plays(plays):
    from playcall.models import assign_callers

    if plays.empty:
        return plays
    context = ui.context()
    return plays.loc[assign_callers(plays, context).eq(context["caller"])]


def matchups():
    context = ui.context()
    st.caption(
        f"Verified playcaller intervals for {context['caller']}, limited to Flores’s Minnesota tenure."
    )
    with st.expander("Playcaller source records", expanded=True):
        st.dataframe(
            context["intervals"],
            width="stretch",
            hide_index=True,
            column_config={
                "source": st.column_config.LinkColumn(
                    "Team announcement", display_text="Open source"
                )
            },
        )
    plays = caller_plays(ui.get_plays())
    if plays.empty:
        st.info("Load the play-by-play data to check the verified intervals for direct games.")
        return
    direct = plays.loc[plays["defteam"].eq("MIN") & plays["season"].ge(2023)]
    if direct.empty:
        st.metric("Verified direct games in local data", 0)
        st.info("No completed qualifying games appear in the loaded play-by-play records.")
        st.caption(
            "The same coach must have called the offense against Minnesota during Flores’s tenure."
        )
        return
    overview = (
        direct.assign(passes=direct["call"].eq("pass"))
        .groupby(["game_id", "game_date", "posteam"], as_index=False)
        .agg(
            plays=("call", "size"), passes=("passes", "sum"), opening_plays=("opening_drive", "sum")
        )
    )
    st.dataframe(overview, hide_index=True, width="stretch")
    game = st.selectbox("Direct game", sorted(direct["game_id"].unique()))
    sample = direct.loc[direct["game_id"].eq(game)]
    opening_only = st.toggle("Opening drive only", value=True)
    if opening_only:
        sample = sample.loc[sample["opening_drive"]]
    if sample.empty:
        st.info("There are no eligible plays in this game selection.")
        return
    cols = st.columns(3)
    cols[0].metric("Eligible plays", len(sample))
    cols[1].metric("Runs", int(sample["call"].eq("run").sum()))
    cols[2].metric("Passes", int(sample["call"].eq("pass").sum()))
    st.subheader("Play sequence")
    fields = [
        "play_id",
        "down",
        "ydstogo",
        "yardline_100",
        "call",
        "yards_gained",
        "offense_personnel",
        "offense_formation",
        "was_pressure",
        "defense_coverage_type",
        "desc",
    ]
    st.dataframe(
        sample[[field for field in fields if field in sample]], width="stretch", hide_index=True
    )
    st.caption(
        "A single game can show an observed tendency. Use the forecast’s uncertainty estimates for predictions."
    )


def similarity():
    model = ui.require_model()
    if model is None:
        return
    profiles = model.defensive_similarity("MIN")
    if profiles.empty:
        st.info("The fitted model has no defensive profile for Minnesota.")
        return
    st.caption(
        "Profiles use historical play-by-play outcomes. They summarize defensive behavior indirectly."
    )
    with st.expander("How the score works"):
        st.write(
            "Pass share faced, quarterback-hit rate, sack rate, pass and run EPA, and yards allowed where available. "
            "Available FTN charting can add blitz and five-or-more-rusher rates. "
            "The model standardizes each measure across teams, then converts its root mean squared distance "
            "from Minnesota to exp(−distance). Identical profiles score 1. The table marks the defenses used by Model B."
        )
    count = st.slider(
        "Defenses shown", min_value=1, max_value=max(2, len(profiles)), value=min(10, len(profiles))
    )
    subset = profiles.head(count)
    fig = px.bar(
        subset.iloc[::-1],
        y="defteam",
        x="similarity",
        orientation="h",
        color_discrete_sequence=[ui.PASS],
        labels={"defteam": "Defense", "similarity": "Profile similarity"},
    )
    ui.figure(fig, height=max(325, count * 24 + 100))
    st.dataframe(subset, width="stretch", hide_index=True)
    defense = st.selectbox("Defense to inspect", subset["defteam"])
    plays = caller_plays(ui.get_plays())
    if plays.empty:
        st.info("Load play-by-play data to inspect offensive behavior against this defense.")
        return
    sample = plays.loc[plays["defteam"].eq(defense)].copy()
    sample["pass"] = sample["call"].eq("pass")
    st.subheader(f"Verified playcaller offense vs {defense}")
    if sample.empty:
        st.info("The local records have no eligible plays for this caller and defense.")
        return
    summary = sample.groupby(["game_date", "posteam"], as_index=False).agg(
        plays=("call", "size"), pass_share=("pass", "mean"), mean_yards=("yards_gained", "mean")
    )
    st.dataframe(
        summary,
        width="stretch",
        hide_index=True,
        column_config={"pass_share": st.column_config.NumberColumn("Pass share", format="percent")},
    )


DIMENSIONS = {
    "Down and distance": ("down", "ydstogo"),
    "Field position": ("yardline_100",),
    "Field hash": ("starting_hash", "field_hash", "hash"),
    "Quarter and clock": ("qtr", "quarter_seconds_remaining"),
    "Stadium": ("stadium", "game_stadium"),
    "Offensive personnel": ("offense_personnel",),
    "Defensive personnel": ("defense_personnel",),
    "Offensive formation": ("offense_formation",),
    "Quarterback alignment": ("qb_location",),
    "Defensive coverage family": ("defense_coverage_type", "defense_man_zone_type"),
    "Previous play sequence": ("previous_call",),
}


def _bucket_labels(frame, dimension):
    columns = DIMENSIONS[dimension]
    if dimension == "Down and distance":
        distances = pd.cut(frame["ydstogo"], [0, 3, 7, float("inf")], labels=["1–3", "4–7", "8+"])
        return frame["down"].astype("Int64").astype("string") + " & " + distances.astype("string")
    if dimension == "Field position":
        return pd.cut(
            frame["yardline_100"],
            [0, 20, 50, 80, 100],
            labels=[
                "Goal to 20",
                "Opponent 21 to midfield",
                "Own 20 to midfield",
                "Own goal to 19",
            ],
        )
    if dimension == "Quarter and clock":
        minutes = pd.cut(
            frame["quarter_seconds_remaining"], [-1, 300, 600, 900], labels=["0–5", "5–10", "10–15"]
        )
        return (
            "Q"
            + frame["qtr"].astype("Int64").astype("string")
            + " · "
            + minutes.astype("string")
            + " min"
        )
    if dimension == "Previous play sequence":
        first = frame.groupby(["game_id", "posteam", "drive"])["previous_call"].shift()
        previous = frame["previous_call"].astype("string").fillna("Start")
        return first.astype("string").str.cat(previous, sep=" → ").fillna(previous)
    for column in columns:
        if column in frame and frame[column].notna().any():
            return frame[column]
    return pd.Series(pd.NA, index=frame.index, dtype="string")


def history():
    st.caption("Compare measured run and pass frequencies for eligible scrimmage plays.")
    plays = ui.get_plays()
    if plays.empty:
        st.info("Refresh the local play-by-play cache to enable these charts.")
        return
    filters = st.columns(3)
    teams = ["All teams", *sorted(plays["posteam"].unique())]
    team = filters[0].selectbox("Offense", teams, index=teams.index("ATL") if "ATL" in teams else 0)
    years = sorted(int(year) for year in plays["season"].unique())
    seasons = filters[1].multiselect("Seasons", years, default=years)
    dimension = filters[2].selectbox("Group by", list(DIMENSIONS))
    opening = st.toggle("Opening drives only")
    sample = plays.loc[plays["season"].isin(seasons)]
    if team != "All teams":
        sample = sample.loc[sample["posteam"].eq(team)]
    if opening:
        sample = sample.loc[sample["opening_drive"]]
    if sample.empty:
        st.info("No eligible plays match these filters.")
        return
    labels = _bucket_labels(sample, dimension)
    counts = (
        sample.assign(group=labels)
        .groupby(["group", "call"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    if counts.empty:
        st.info(f"{dimension} has no observed labels in this selection.")
        return
    totals = counts.groupby("group", observed=True)["n"].sum().sort_values(ascending=False)
    groups = totals.head(18).index.tolist()
    plotted = counts.loc[counts["group"].isin(groups)].copy()
    plotted["share"] = plotted["n"] / plotted["group"].astype("string").map(totals)
    st.caption(
        f"{int(totals.sum()):,} labeled of {len(sample):,} eligible plays · {len(groups)} of {len(totals)} groups shown"
    )
    fig = px.bar(
        plotted,
        x="share",
        y="group",
        color="call",
        orientation="h",
        custom_data=["n"],
        category_orders={"group": groups, "call": ["run", "pass"]},
        color_discrete_map={"run": ui.RUN, "pass": ui.PASS},
        labels={"share": "Share of eligible plays", "group": dimension, "call": "Playcall"},
    )
    fig.update_traces(hovertemplate="%{y}<br>%{x:.1%} · %{customdata[0]:,} plays")
    fig.update_xaxes(tickformat=".0%", range=[0, 1])
    fig.update_yaxes(autorange="reversed")
    ui.figure(fig, height=max(340, len(groups) * 29 + 130))
    st.dataframe(
        plotted[["group", "call", "n", "share"]],
        width="stretch",
        hide_index=True,
        column_config={"share": st.column_config.NumberColumn("Share", format="percent")},
    )

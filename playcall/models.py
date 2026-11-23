"""Opening-drive playcall models and chronological evaluation."""

import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from playcall.data import _atomic_json

SEED = 85029
CONTEXT_PATH = Path(__file__).with_name("context.json")
DEFAULT_MODEL_PATH = "models/playcall.joblib"
DEFAULT_REPORT_PATH = "models/backtest.json"
NUMERIC = ["down", "ydstogo", "yardline_100", "score_differential", "game_seconds_remaining"]
CONTEXT = ["offense_rest", "defense_rest", "is_home"]
CATEGORICAL = ["posteam", "defteam", "previous_call", "caller"]
SITUATION = ["down", "distance_band", "field_band"]
PROFILE_FEATURES = [
    "pass_share",
    "qb_hit_rate",
    "sack_rate",
    "pass_epa",
    "run_epa",
    "yards_allowed",
]
CHARTED_PROFILE_FEATURES = ["chart_blitz_rate", "chart_five_plus_rushers_rate"]
TACTICAL = {
    "offense_personnel": "Personnel grouping",
    "offense_formation": "Formation family",
    "shotgun": "Shotgun indicator",
    "qb_location": "Charted quarterback alignment",
    "is_motion": "Pre-snap motion",
    "is_play_action": "Play action",
    "is_rpo": "Run-pass option",
    "is_screen_pass": "Screen pass",
    "run_location": "Run direction",
    "pass_location": "Pass target direction",
    "pass_length": "Pass target depth",
    "route": "Charted primary receiver route",
    "defense_man_zone_type": "Charted man or zone",
}
TACTICAL_LABELS = {
    "shotgun": {True: "Shotgun", False: "Non-shotgun"},
    "qb_location": {"U": "Under center", "S": "Shotgun", "P": "Pistol"},
    "is_motion": {True: "Motion", False: "No motion"},
    "is_play_action": {True: "Play action", False: "No play action"},
    "is_rpo": {True: "RPO", False: "No RPO"},
    "is_screen_pass": {True: "Screen", False: "No screen"},
}


def load_context(path=CONTEXT_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def assign_callers(frame, context):
    """Keep unverified caller assignments unknown, even when a head coach is listed."""
    caller = pd.Series("Unknown", index=frame.index, dtype="object")
    for interval in context["intervals"]:
        mask = (
            frame["posteam"].eq(interval["team"])
            & frame["season"].eq(interval["season"])
            & frame["week"].between(interval["min_week"], interval["max_week"])
        )
        caller.loc[mask] = interval.get("caller", context["caller"])
    return caller


def _situation(frame):
    return frame.assign(
        distance_band=np.digitize(frame["ydstogo"], [3, 7, 11]),
        field_band=np.digitize(frame["yardline_100"], [21, 51, 81]),
    )


def _rate_table(frame, keys):
    return frame.groupby(keys, observed=True)["target"].agg(["sum", "count"])


def _lookup(frame, table, keys):
    index = pd.MultiIndex.from_frame(frame[keys]) if len(keys) > 1 else pd.Index(frame[keys[0]])
    return table.reindex(index, fill_value=0).to_numpy(dtype=float).T


def _prior_probability(hits, trials, prior, strength):
    return (hits + strength * prior) / (trials + strength)


def _profile_table(frame, availability_cutoff=None):
    data = frame.copy()
    data = data.loc[~data["defteam"].eq("MIN") | data["season"].ge(2023)]
    # Use only played games in the fit slice, with equal emphasis per play in each window.
    rank = data[["defteam", "game_id", "game_date"]].drop_duplicates()
    rank = rank.sort_values(["game_date", "game_id"], ascending=False)
    rank = rank.loc[rank.groupby("defteam").cumcount().lt(32)]
    data = data.merge(
        rank[["defteam", "game_id"]], on=["defteam", "game_id"], validate="many_to_one"
    )
    pass_play = data["target"].eq(1)
    data["pass_epa"] = data["epa"].where(pass_play)
    data["run_epa"] = data["epa"].where(~pass_play)
    data["qb_hit_rate"] = data["qb_hit"].where(pass_play)
    data["sack_rate"] = data["sack"].where(pass_play)
    for column, rate, threshold in [
        ("n_blitzers", "chart_blitz_rate", 1),
        ("n_pass_rushers", "chart_five_plus_rushers_rate", 5),
    ]:
        data[rate] = np.nan
        if column in data and "charting_date_pulled" in data and availability_cutoff is not None:
            eligible = pd.to_datetime(data["charting_date_pulled"], utc=True).lt(
                availability_cutoff
            )
            observed = data[column].where(pass_play & eligible)
            data[rate] = observed.ge(threshold).where(observed.notna()).astype(float)
    return data.groupby("defteam").agg(
        pass_share=("target", "mean"),
        qb_hit_rate=("qb_hit_rate", "mean"),
        sack_rate=("sack_rate", "mean"),
        pass_epa=("pass_epa", "mean"),
        run_epa=("run_epa", "mean"),
        yards_allowed=("yards_gained", "mean"),
        chart_blitz_rate=("chart_blitz_rate", "mean"),
        chart_blitz_n=("chart_blitz_rate", "count"),
        chart_five_plus_rushers_rate=("chart_five_plus_rushers_rate", "mean"),
        chart_pass_rush_n=("chart_five_plus_rushers_rate", "count"),
        plays=("target", "size"),
        games=("game_id", "nunique"),
    )


def _scaled_profiles(profile):
    coverage_columns = ["chart_blitz_n", "chart_pass_rush_n"]
    usable = (
        set(coverage_columns).issubset(profile.columns)
        and profile[coverage_columns].ge(24).all().all()
    )
    fields = PROFILE_FEATURES + CHARTED_PROFILE_FEATURES if usable else PROFILE_FEATURES
    scaled = profile[fields]
    scaled = (scaled - scaled.mean()) / scaled.std().replace(0, 1).fillna(1)
    return scaled.fillna(0)


def _similarity(profile):
    scaled = _scaled_profiles(profile).to_numpy()
    distances = np.sqrt(np.mean((scaled[:, None, :] - scaled[None, :, :]) ** 2, axis=2))
    affinity = pd.DataFrame(np.exp(-distances), index=profile.index, columns=profile.index)
    np.fill_diagonal(affinity.values, 0)
    # Exclude weak analogues and distinguish the target from the analogues.
    return affinity.where(affinity.rank(axis=1, method="first", ascending=False).le(8), 0)


class RateModels:
    def __init__(self, all_plays, opening, context, availability_cutoff=None):
        self.caller = context["caller"]
        self.league = float(opening["target"].mean())
        self.situational = _rate_table(opening, SITUATION)
        self.by_caller = _rate_table(
            opening.loc[opening["caller"].ne("Unknown")], ["caller", *SITUATION]
        )
        direct = opening.loc[
            opening["caller"].eq(self.caller)
            & opening["defteam"].eq("MIN")
            & opening["season"].ge(2023)
        ]
        self.direct = _rate_table(direct, SITUATION)
        self.direct_count = len(direct)
        self.profiles = _profile_table(all_plays, availability_cutoff)
        self.affinity = _similarity(self.profiles)
        # Recent windows are per team and end at the common fitting cutoff.
        latest_offense = opening[["posteam", "game_id", "game_date"]].drop_duplicates()
        latest_defense = opening[["defteam", "game_id", "game_date"]].drop_duplicates()
        recent = []
        for group, latest in [("posteam", latest_offense), ("defteam", latest_defense)]:
            latest = latest.sort_values(["game_date", "game_id"], ascending=False)
            latest = latest.loc[latest.groupby(group).cumcount().lt(10)]
            selected = opening.merge(latest[[group, "game_id"]], on=[group, "game_id"])
            recent.append(_rate_table(selected, [group, *SITUATION]))
        self.recent_offense, self.recent_defense = recent
        self.analogues = []
        for identity in ["posteam", "caller"]:
            sample = (
                opening if identity == "posteam" else opening.loc[opening["caller"].ne("Unknown")]
            )
            table = _rate_table(sample, [identity, *SITUATION, "defteam"]).unstack(fill_value=0)
            self.analogues.append(table)

    def predict(self, frame):
        hits, count = _lookup(frame, self.situational, SITUATION)
        situational = _prior_probability(hits, count, self.league, 30)
        caller_hits, caller_count = _lookup(frame, self.by_caller, ["caller", *SITUATION])
        caller_p = _prior_probability(caller_hits, caller_count, situational, 24)
        off_hits, off_n = _lookup(frame, self.recent_offense, ["posteam", *SITUATION])
        def_hits, def_n = _lookup(frame, self.recent_defense, ["defteam", *SITUATION])
        # Defense outcomes mix opponents. Count them at half the effective weight.
        recent_hits, recent_n = (
            off_hits + 0.5 * def_hits + caller_hits,
            off_n + 0.5 * def_n + caller_count,
        )
        recent_p = _prior_probability(recent_hits, recent_n, situational, 24)
        direct_hits, direct_n = _lookup(frame, self.direct, SITUATION)
        verified_direct = frame["defteam"].eq("MIN") & frame["caller"].eq(self.caller)
        direct_n = np.where(verified_direct, direct_n, 0)
        direct_hits = np.where(verified_direct, direct_hits, 0)
        direct_p = _prior_probability(direct_hits, direct_n, recent_p, 24)
        weights = self.affinity.reindex(index=frame["defteam"], fill_value=0).to_numpy()
        analogue_stats = []
        for identity, table in zip(["posteam", "caller"], self.analogues, strict=True):
            index = pd.MultiIndex.from_frame(frame[[identity, *SITUATION]])
            groups = table.reindex(index, fill_value=0)
            counts = []
            for column in ("sum", "count"):
                defense_columns = pd.MultiIndex.from_product([[column], self.affinity.columns])
                per_defense = groups.reindex(columns=defense_columns, fill_value=0).to_numpy()
                counts.append(np.einsum("ij,ij->i", per_defense, weights))
            analogue_stats.append(counts)
        (team_hits, team_n), (caller_analogue_hits, caller_analogue_n) = analogue_stats
        analog_hits = np.where(caller_analogue_n > 0, caller_analogue_hits, team_hits)
        analog_n = np.where(caller_analogue_n > 0, caller_analogue_n, team_n)
        analog_p = _prior_probability(analog_hits, analog_n, recent_p, 24)
        return {
            "league": np.full(len(frame), self.league),
            "situational": situational,
            "recent_offense": _prior_probability(off_hits, off_n, situational, 24),
            "historical_caller": caller_p,
            "A": direct_p,
            "B": analog_p,
            "C": recent_p,
            "A_n": direct_n,
            "B_n": analog_n,
            "C_n": recent_n,
            "B_caller": caller_analogue_n > 0,
        }


def _ensemble(rates, analog_weight, direct_priority=0.65):
    # Component A receives the strongest initial prior, tempered by actual support.
    weight_a = direct_priority * rates["A_n"] / (rates["A_n"] + 24)
    weight_b = (1 - weight_a) * analog_weight * rates["B_n"] / (rates["B_n"] + 12)
    weight_c = 1 - weight_a - weight_b
    return weight_a * rates["A"] + weight_b * rates["B"] + weight_c * rates["C"], (
        weight_a,
        weight_b,
        weight_c,
    )


def _ml_pipeline(kind):
    numeric = NUMERIC if kind == "logistic_situation" else NUMERIC + CONTEXT
    categorical = [] if kind == "logistic_situation" else CATEGORICAL
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
                ),
                numeric,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )
    if kind.startswith("logistic"):
        estimator = LogisticRegression(C=0.35, max_iter=350, random_state=SEED)
    elif kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=140,
            max_depth=12,
            min_samples_leaf=24,
            random_state=SEED,
            n_jobs=4,
        )
    else:
        from xgboost import XGBClassifier

        estimator = XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.85,
            colsample_bytree=0.9,
            random_state=SEED,
            n_jobs=4,
            tree_method="hist",
        )
    return make_pipeline(preprocessing, estimator)


def _fit_candidates(all_plays, opening, context, availability_cutoff=None):
    kinds = ["logistic", "logistic_situation", "random_forest"]
    try:
        import xgboost  # noqa: F401
    except ImportError:
        pass
    else:
        kinds.append("xgboost")
    models = {kind: _ml_pipeline(kind).fit(opening, opening["target"]) for kind in kinds}
    return RateModels(all_plays, opening, context, availability_cutoff), models


def _candidate_probabilities(models, rates, frame, analog_weight):
    components = rates.predict(frame)
    predictions = {
        key: value
        for key, value in components.items()
        if key not in {"A_n", "B_n", "C_n", "B_caller"}
    }
    predictions["ensemble"] = _ensemble(components, analog_weight)[0]
    predictions["ensemble_without_similarity"] = _ensemble(components, 0)[0]
    predictions.update({name: model.predict_proba(frame)[:, 1] for name, model in models.items()})
    return predictions


def _logits(probabilities):
    clipped = np.clip(probabilities, 1e-5, 1 - 1e-5)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def _apply_calibration(calibrator, probabilities):
    return (
        calibrator.predict_proba(_logits(probabilities))[:, 1]
        if calibrator is not None
        else probabilities
    )


def _metrics(frame, probabilities):
    labels = frame["target"].to_numpy()
    bins = np.minimum((10 * probabilities).astype(int), 9)
    reliability = []
    for index in np.unique(bins):
        mask = bins == index
        reliability.append(
            {
                "bin": int(index),
                "n": int(mask.sum()),
                "predicted_pass": float(probabilities[mask].mean()),
                "observed_pass": float(labels[mask].mean()),
            }
        )
    drive = (
        frame.assign(prediction=probabilities)
        .groupby(["game_id", "posteam"])[["prediction", "target"]]
        .mean()
    )
    ece = sum(row["n"] * abs(row["predicted_pass"] - row["observed_pass"]) for row in reliability)
    return {
        "n": len(frame),
        "accuracy": float(accuracy_score(labels, probabilities >= 0.5)),
        "top2_accuracy": 1.0,
        "log_loss": float(log_loss(labels, probabilities, labels=[0, 1])),
        "brier": float(brier_score_loss(labels, probabilities)),
        "calibration_error": float(ece / len(frame)),
        "drive_pass_share_mae": float((drive["prediction"] - drive["target"]).abs().mean()),
        "reliability": reliability,
    }


def _importance(ml, rates, selected, frame, calibrator, analog_weight):
    features = NUMERIC + CONTEXT + CATEGORICAL
    rng = np.random.default_rng(SEED)

    def loss(inputs):
        if selected in ml:
            probability = ml[selected].predict_proba(inputs)[:, 1]
        else:
            prepared = _situation(inputs)
            components = rates.predict(prepared)
            probability = (
                _ensemble(components, analog_weight)[0]
                if selected == "ensemble"
                else components[selected]
            )
        return log_loss(frame["target"], _apply_calibration(calibrator, probability))

    baseline = loss(frame)
    importance = []
    for feature in features:
        scores = []
        permuted = frame.copy()
        for _ in range(3):
            permuted[feature] = rng.permutation(frame[feature].to_numpy())
            scores.append(loss(permuted) - baseline)
        importance.append({"feature": feature, "log_loss_increase": float(np.mean(scores))})
    return sorted(importance, key=lambda row: row["log_loss_increase"], reverse=True)


def _simulate_frame(state, context):
    item = asdict(state) if is_dataclass(state) else dict(state)
    offense, defense = item.get("offense", "ATL"), item.get("defense", "MIN")
    caller = item.get("caller")
    if caller is None:
        caller = context["caller"] if offense == context["game"]["away"] else "Unknown"
    return _situation(
        pd.DataFrame(
            [
                {
                    "down": item.get("down", 1),
                    "ydstogo": item.get("distance", item.get("ydstogo", 10)),
                    "yardline_100": item.get("yardline_100", 75),
                    "score_differential": item.get("score_differential", 0),
                    "game_seconds_remaining": item.get("game_seconds_remaining", 3600),
                    "offense_rest": item.get("offense_rest", np.nan),
                    "defense_rest": item.get("defense_rest", np.nan),
                    "is_home": bool(item.get("is_home", False)),
                    "posteam": offense,
                    "defteam": defense,
                    "previous_call": item.get("previous_call") or "none",
                    "caller": caller,
                }
            ]
        )
    )


def _local_sample(opening, frame, minimum):
    state = frame.iloc[0]
    same_down_distance = opening["down"].eq(state["down"]) & opening["distance_band"].eq(
        state["distance_band"]
    )
    same_offense = opening["posteam"].eq(state["posteam"])
    caller = opening["caller"].eq(state["caller"]) & opening["caller"].ne("Unknown")
    candidate = opening.loc[same_down_distance & caller]
    if len(candidate) >= minimum:
        return candidate, "verified caller in matching down and distance"
    candidate = opening.loc[same_down_distance & same_offense]
    if len(candidate) >= minimum:
        return candidate, "same offense in matching down and distance"
    return opening.loc[same_down_distance], "league in matching down and distance"


def _tactical_distributions(sample):
    distributions = {}
    for label, description in TACTICAL.items():
        if label not in sample:
            continue
        selected = sample
        if label == "run_location":
            selected = sample.loc[sample["call"].eq("run")]
        if label in {"pass_location", "pass_length", "route", "is_screen_pass"}:
            selected = sample.loc[sample["call"].eq("pass")]
        values = selected[label].dropna()
        if values.empty:
            continue
        if label in TACTICAL_LABELS:
            values = values.map(TACTICAL_LABELS[label])
        distribution = values.astype(str).value_counts(normalize=True)
        if not distribution.empty:
            distributions[label] = {
                "label": description,
                "n": len(values),
                "coverage": len(values) / len(selected),
                "probabilities": distribution.to_dict(),
            }
    return distributions


class PlaycallModels:
    def __init__(self, rates, ml, opening, calibrator, report, context):
        self.rates = rates
        self.ml = ml
        self.opening = opening
        self.calibrator = calibrator
        self.report = report
        self.context = context
        self.version = report["model_version"]
        self.dataset_version = report["dataset_version"]

    def defensive_similarity(self, defense="MIN"):
        table = self.rates.profiles.copy()
        if defense not in table.index:
            return table.iloc[:0]
        # Display all similarity scores. Model B applies the top-eight restriction.
        scaled = _scaled_profiles(table)
        distance = ((scaled - scaled.loc[defense]).pow(2).mean(axis=1)).pow(0.5)
        table["similarity"] = np.exp(-distance)
        table["used_by_model_b"] = self.rates.affinity.loc[defense].gt(0)
        return table.sort_values("similarity", ascending=False).reset_index()

    def direct_matchups(self):
        return self.opening.loc[
            self.opening["caller"].eq(self.context["caller"])
            & self.opening["defteam"].eq("MIN")
            & self.opening["season"].ge(2023)
        ].copy()

    def call_probabilities(self, state):
        frame = _simulate_frame(state, self.context)
        rates = self.rates.predict(frame)
        weight = self.report["ensemble_analog_weight"]
        ensemble, weights = _ensemble(rates, weight)
        name = self.report["selected_model"]
        if name == "ensemble":
            raw = ensemble
        elif name in rates:
            raw = rates[name]
        else:
            raw = self.ml[name].predict_proba(frame)[:, 1]
        p_pass = float(_apply_calibration(self.calibrator, raw)[0])
        components = {
            key: {
                "p_pass": float(rates[key][0]),
                "p_run": 1 - float(rates[key][0]),
                "effective_n": float(rates[f"{key}_n"][0]),
                "weight": float(fraction[0]),
            }
            for key, fraction in zip(("A", "B", "C"), weights, strict=True)
        }
        components["B"]["sample_basis"] = (
            "verified caller" if rates["B_caller"][0] else "same offense proxy"
        )
        return {
            "p_pass": p_pass,
            "p_run": 1 - p_pass,
            "components": components,
            "ensemble_p_pass": float(ensemble[0]),
            "selected_model": name,
        }

    def predict(self, state):
        probability = self.call_probabilities(state)
        frame = _simulate_frame(state, self.context)
        sample, basis = _local_sample(self.opening, frame, minimum=15)
        personnel = (
            asdict(state).get("personnel") if is_dataclass(state) else state.get("personnel")
        )
        if personnel and "offense_personnel" in sample:
            personnel_subset = sample.loc[sample["offense_personnel"].eq(personnel)]
            if len(personnel_subset) >= 15:
                sample, basis = personnel_subset, f"{basis}, with the selected personnel"
        n = len(sample)
        # Wilson interval describes the empirical reference sample, never the model probability.
        reference = float(sample["target"].mean()) if n else None
        interval = None
        if n:
            denominator = 1 + 1.96**2 / n
            midpoint = (reference + 1.96**2 / (2 * n)) / denominator
            radius = (
                1.96 * np.sqrt((reference * (1 - reference) + 1.96**2 / (4 * n)) / n) / denominator
            )
            interval = [float(midpoint - radius), float(midpoint + radius)]
        probability.update(
            tactical=_tactical_distributions(sample),
            uncertainty={
                "n": n,
                "basis": basis,
                "empirical_pass_rate": reference,
                "wilson95": interval,
            },
        )
        return probability

    def outcome_distribution(self, state, call):
        """Reference outcomes conditional on a scrimmage call, pooled conservatively."""
        frame = _simulate_frame(state, self.context)
        sample, _ = _local_sample(
            self.opening.loc[self.opening["call"].eq(call)], frame, minimum=24
        )
        if sample.empty:
            sample = self.opening.loc[self.opening["call"].eq(call)]
        state_values = frame.iloc[0]
        yards = sample["yards_gained"].fillna(0).astype(int)
        turnover = sample["interception"].fillna(0).eq(1) | sample["fumble_lost"].fillna(0).eq(1)
        # Exact gain is retained at transitions. The tree prunes less likely values.
        outcome = pd.DataFrame(
            {
                "yards": yards,
                "turnover": turnover,
                "touchdown": yards.ge(state_values["yardline_100"]),
                "safety": yards.le(state_values["yardline_100"] - 100),
            }
        )
        frequencies = outcome.value_counts(normalize=True, sort=False).sort_values(ascending=False)
        return [
            {
                "yards": int(value[0]),
                "turnover": bool(value[1]),
                "touchdown": bool(value[2]),
                "safety": bool(value[3]),
                "probability": float(weight),
            }
            for value, weight in frequencies.items()
        ]


def _git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _prepare(store, cutoff, context):
    frame = store.plays()
    if frame.empty:
        raise ValueError("Refresh nflverse data before training.")
    frame = frame.loc[pd.to_datetime(frame["game_date"]).le(pd.Timestamp(cutoff))].copy()
    frame["caller"] = assign_callers(frame, context)
    frame["target"] = frame["call"].eq("pass").astype(int)
    frame["previous_call"] = frame["previous_call"].fillna("none")
    for name in [*CONTEXT, "epa", "qb_hit", "sack", "interception", "fumble_lost"]:
        if name not in frame:
            frame[name] = np.nan
    return _situation(frame)


def chronological_split(frame):
    seasons = sorted(frame["season"].unique())
    if len(seasons) < 3:
        raise ValueError(
            "At least three distinct cached NFL seasons are required for train, validation and test."
        )
    test_count = 2 if len(seasons) >= 5 else 1
    validation = seasons[-(test_count + 1)]
    split = (
        frame["season"].lt(validation),
        frame["season"].eq(validation),
        frame["season"].gt(validation),
    )
    return tuple(frame.loc[mask] for mask in split)


def fit_models(store, model_path=DEFAULT_MODEL_PATH, report_path=DEFAULT_REPORT_PATH, cutoff=None):
    """Train for the target date and evaluate a separate, frozen chronological split."""
    context = load_context()
    cutoff = pd.Timestamp(cutoff or context["as_of"]).date().isoformat()
    frame = _prepare(store, cutoff, context)
    fit, validation, test = chronological_split(frame)
    train_open, validation_open, test_open = [
        item.loc[item["opening_drive"]] for item in (fit, validation, test)
    ]
    if not all(
        len(item) and item["target"].nunique() == 2
        for item in (train_open, validation_open, test_open)
    ):
        raise ValueError("Each chronological split requires opening-drive plays from both classes.")
    val_cutoff = pd.Timestamp(validation_open["game_date"].min(), tz="UTC")
    rates, ml = _fit_candidates(fit, train_open, context, val_cutoff)
    validation_rates = rates.predict(validation_open)
    analog_weight = min(
        np.linspace(0, 1, 11),
        key=lambda weight: log_loss(
            validation_open["target"], _ensemble(validation_rates, weight)[0]
        ),
    )
    validation_prob = _candidate_probabilities(ml, rates, validation_open, analog_weight)
    test_prob = _candidate_probabilities(ml, rates, test_open, analog_weight)
    rankable = [
        "league",
        "situational",
        "recent_offense",
        "historical_caller",
        "ensemble",
        "logistic",
        "logistic_situation",
        "random_forest",
        "xgboost",
    ]
    calibrators = {}
    validation_loss = {}
    metrics = {}
    for name, probabilities in validation_prob.items():
        # Select model and calibration using the validation slice only.
        raw_loss = log_loss(validation_open["target"], probabilities)
        calibrator = LogisticRegression(C=10, random_state=SEED).fit(
            _logits(probabilities), validation_open["target"]
        )
        calibrated_loss = log_loss(
            validation_open["target"], _apply_calibration(calibrator, probabilities)
        )
        calibrators[name] = calibrator if raw_loss > calibrated_loss else None
        validation_loss[name] = float(min(raw_loss, calibrated_loss))
        calibrated_test = _apply_calibration(calibrators[name], test_prob[name])
        metrics[name] = _metrics(test_open, calibrated_test)
        metrics[name]["raw_log_loss"] = float(log_loss(test_open["target"], test_prob[name]))
        metrics[name]["validation_log_loss"] = validation_loss[name]
        metrics[name]["calibration"] = "sigmoid" if calibrators[name] is not None else "identity"
    selected = min((name for name in rankable if name in validation_loss), key=validation_loss.get)
    atlanta_mask = test_open["posteam"].eq(context["game"]["away"]).to_numpy()
    atlanta_metrics = None
    if atlanta_mask.any():
        atlanta_prob = _apply_calibration(calibrators[selected], test_prob[selected][atlanta_mask])
        atlanta_metrics = _metrics(test_open.loc[atlanta_mask], atlanta_prob)
    # Isolate the value of charted fields with identical pre-test games and fixed weights.
    earlier = pd.concat([fit, validation], ignore_index=True)
    earlier_opening = earlier.loc[earlier["opening_drive"]]
    test_cutoff = pd.Timestamp(test_open["game_date"].min(), tz="UTC")
    similarity_ablation = {}
    for basis, available in [("play_by_play", None), ("available_ftn", test_cutoff)]:
        comparison = RateModels(earlier, earlier_opening, context, available)
        components = comparison.predict(test_open)
        similarity_ablation[basis] = {
            "profile_features": list(_scaled_profiles(comparison.profiles).columns),
            "minimum_charted_dropbacks": int(comparison.profiles["chart_blitz_n"].min()),
            "model_B_log_loss": float(log_loss(test_open["target"], components["B"])),
            "model_C_log_loss": float(log_loss(test_open["target"], components["C"])),
            "fixed_ensemble_log_loss": float(
                log_loss(test_open["target"], _ensemble(components, 0.5)[0])
            ),
        }
    dataset_version = store.manifest()["dataset_version"]
    commit = _git_commit()
    model_version = sha256(
        f"{dataset_version}:{commit}:{cutoff}:{selected}:{SEED}".encode()
    ).hexdigest()[:16]
    report = {
        "model_version": model_version,
        "dataset_version": dataset_version,
        "git_commit": commit,
        "generated_at": datetime.now(UTC).isoformat(),
        "training_cutoff": cutoff,
        "seed": SEED,
        "selected_model": selected,
        "ensemble_analog_weight": float(analog_weight),
        "metrics": metrics,
        "atlanta_selected_metrics": atlanta_metrics,
        "feature_importance": _importance(
            ml, rates, selected, test_open, calibrators[selected], analog_weight
        ),
        "similarity_ablation": similarity_ablation,
        "splits": {
            key: {
                "start": str(item["game_date"].min()),
                "end": str(item["game_date"].max()),
                "n": len(item),
                "games": int(item["game_id"].nunique()),
            }
            for key, item in [
                ("train", train_open),
                ("validation", validation_open),
                ("test", test_open),
            ]
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": sklearn_version,
            "device": "CPU",
        },
        "notes": [
            "Top-two accuracy is 1 by definition for two classes.",
            "The test years are never used for model or ensemble selection.",
            "Defensive similarity adds FTN blitz and five-plus rush rates only if each defense has at least "
            "24 eligible charted dropbacks; source date_pulled must precede the snapshot cutoff.",
            "Similarity ablation uses a separate pre-test fit through the validation season and a fixed "
            "50% maximum analog weight before sample-size shrinkage.",
            "Production predictors are refit through training_cutoff; validation sigmoid parameters are retained.",
        ],
    }
    opening = frame.loc[frame["opening_drive"]].copy()
    ftn_cutoff = min(
        pd.Timestamp(cutoff, tz="UTC") + pd.Timedelta(days=1), pd.Timestamp.now(tz="UTC")
    )
    rates, ml = _fit_candidates(frame, opening, context, ftn_cutoff)
    model = PlaycallModels(rates, ml, opening, calibrators[selected], report, context)
    model_path, report_path = Path(model_path), Path(report_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=model_path.parent, delete=False) as stream:
        temporary = Path(stream.name)
    try:
        joblib.dump(model, temporary, compress=3)
        temporary.replace(model_path)
    finally:
        temporary.unlink(missing_ok=True)
    _atomic_json(report_path, report)
    return report


def load_models(path=DEFAULT_MODEL_PATH):
    """Load a checkpoint created locally by this project."""
    return joblib.load(Path(path))

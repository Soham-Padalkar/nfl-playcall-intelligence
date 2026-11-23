"""Incremental nflverse downloads and DuckDB queries over Parquet."""

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import duckdb
import pandas as pd
import pyarrow.parquet as pq
import requests

RELEASES = "https://github.com/nflverse/nflverse-data/releases/download"
SCHEDULES = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_REQUIRED = {
    "game_id",
    "play_id",
    "season",
    "week",
    "game_date",
    "play_type",
    "posteam",
    "defteam",
    "drive",
    "down",
    "ydstogo",
    "yardline_100",
    "game_seconds_remaining",
    "score_differential",
    "yards_gained",
    "qb_dropback",
}
PBP_OPTIONAL = {
    "season_type",
    "qtr",
    "quarter_seconds_remaining",
    "half_seconds_remaining",
    "game_half",
    "fixed_drive",
    "shotgun",
    "no_huddle",
    "qb_scramble",
    "qb_kneel",
    "qb_spike",
    "pass",
    "run",
    "air_yards",
    "pass_length",
    "pass_location",
    "run_location",
    "run_gap",
    "passer_player_id",
    "passer_player_name",
    "rusher_player_id",
    "rusher_player_name",
    "receiver_player_id",
    "receiver_player_name",
    "epa",
    "qb_hit",
    "sack",
    "interception",
    "fumble_lost",
    "turnover",
    "home_team",
    "away_team",
    "posteam_type",
    "roof",
    "surface",
    "temp",
    "wind",
    "stadium",
    "weather",
    "game_stadium",
    "touchdown",
    "safety",
    "desc",
}
PARTICIPATION = {
    "offense_formation",
    "offense_personnel",
    "defense_personnel",
    "defenders_in_box",
    "number_of_pass_rushers",
    "was_pressure",
    "route",
    "defense_man_zone_type",
    "defense_coverage_type",
    "players_on_play",
    "offense_players",
    "defense_players",
}
CHARTING = {
    "n_offense_backfield",
    "is_motion",
    "is_play_action",
    "is_screen_pass",
    "n_blitzers",
    "n_pass_rushers",
    "starting_hash",
    "qb_location",
    "n_defense_box",
    "is_rpo",
    "date_pulled",
}


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    required: frozenset[str]
    optional: bool = False
    available: str = "Publication time varies"


def sources(seasons, supplemental=True):
    yield Source(
        "schedules",
        SCHEDULES,
        frozenset({"game_id", "season", "gameday", "home_team", "away_team"}),
        available="Schedule updatable before kickoff. Final scores and observed weather are retrospective.",
    )
    for season in sorted({int(value) for value in seasons}):
        if season < 1999 or season > datetime.now(UTC).year:
            raise ValueError(f"Season must be between 1999 and the current year: {season}")
        yield Source(
            f"pbp_{season}",
            f"{RELEASES}/pbp/play_by_play_{season}.parquet",
            frozenset(PBP_REQUIRED),
            available="After the game. May be revised.",
        )
        if not supplemental:
            continue
        for name, tag, filename, first_season, required, availability in [
            (
                "participation",
                "pbp_participation",
                "pbp_participation",
                2016,
                {"nflverse_game_id", "play_id"},
                "FTN data from 2023 is released after the postseason.",
            ),
            (
                "charting",
                "ftn_charting",
                "ftn_charting",
                2022,
                {"nflverse_game_id", "nflverse_play_id"},
                "FTN charting updates throughout the season. Use source date_pulled for historical eligibility.",
            ),
            (
                "injuries",
                "injuries",
                "injuries",
                2009,
                {"season", "week", "team"},
                "Reports vary by week and may be revised.",
            ),
            (
                "rosters",
                "rosters",
                "roster",
                1920,
                {"season", "team"},
                "Season rosters update as published. Use timestamped depth charts for as-of rankings.",
            ),
            (
                "depth",
                "depth_charts",
                "depth_charts",
                2001,
                ({"dt", "team", "pos_rank"} if season >= 2025 else {"season"}),
                "From 2025 updates have source timestamps. Earlier charts are weekly.",
            ),
        ]:
            if season >= first_season:
                yield Source(
                    f"{name}_{season}",
                    f"{RELEASES}/{tag}/{filename}_{season}.parquet",
                    frozenset(required),
                    optional=True,
                    available=availability,
                )


def _atomic_json(path, payload):
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, indent=2, sort_keys=True)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def validate_parquet(path, required):
    """Check source compatibility before replacing the current cache."""
    parquet = pq.ParquetFile(path)
    missing = set(required).difference(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"{path.name}: missing columns {', '.join(sorted(missing))}")
    if parquet.metadata.num_rows == 0:
        raise ValueError(f"{path.name}: empty dataset")
    return parquet.metadata.num_rows


class DataStore:
    def __init__(self, root="data"):
        self.root = Path(root)
        self.raw = self.root / "raw"

    def _refresh_source(self, source, force):
        path = self.raw / f"{source.name}.parquet"
        sidecar = self.raw / f"{source.name}.json"
        record = _read_json(sidecar)
        now = datetime.now(UTC)
        checked = record.get("checked_at")
        if not force and checked and now - datetime.fromisoformat(checked) < timedelta(hours=6):
            if path.is_file() or record.get("status") == "unavailable":
                return {**record, "status": "cached"} if path.is_file() else record
        headers = {}
        etag = record.get("etag")
        if etag and path.is_file() and not force:
            headers["If-None-Match"] = etag
        temporary = None
        try:
            with requests.get(
                source.url, headers=headers, stream=True, timeout=(15, 90)
            ) as response:
                if response.status_code == 304 and path.is_file():
                    record.update(checked_at=now.isoformat(), status="cached")
                elif response.status_code == 404 and source.optional:
                    record.update(
                        status="cached" if path.is_file() else "unavailable",
                        checked_at=now.isoformat(),
                        url=source.url,
                        availability=source.available,
                    )
                else:
                    response.raise_for_status()
                    digest = sha256()
                    with NamedTemporaryFile(dir=self.raw, delete=False) as stream:
                        temporary = Path(stream.name)
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            stream.write(chunk)
                            digest.update(chunk)
                    if source.name == "schedules":
                        frame = pd.read_csv(temporary, low_memory=False)
                        frame.to_parquet(temporary, index=False)
                    row_count = validate_parquet(temporary, source.required)
                    fingerprint = digest.hexdigest()
                    changed = fingerprint != record.get("sha256") or not path.is_file()
                    if changed:
                        temporary.replace(path)
                    record = {
                        "status": "updated" if changed else "cached",
                        "url": source.url,
                        "sha256": fingerprint,
                        "rows": row_count,
                        "checked_at": now.isoformat(),
                        "retrieved_at": now.isoformat() if changed else record.get("retrieved_at"),
                        "epoch": response.headers.get("Last-Modified"),
                        "etag": response.headers.get("ETag"),
                        "availability": source.available,
                    }
            _atomic_json(sidecar, record)
        except (requests.RequestException, ValueError, OSError) as error:
            # Request exception strings may contain expiring redirect signatures.
            record.update(
                status="stale" if path.is_file() else "error",
                url=source.url,
                error=type(error).__name__,
                availability=source.available,
            )
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
        return record

    def refresh(self, seasons, force=False, supplemental=True):
        """Refresh requested seasons without discarding other cached seasons."""
        plan = list(sources(seasons, supplemental))
        self.raw.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda source: self._refresh_source(source, force), plan))
        failures = [
            source.name
            for source, record in zip(plan, results, strict=True)
            if not source.optional and record["status"] == "error"
        ]
        if failures:
            raise RuntimeError(
                f"Required nflverse sources could not be loaded: {', '.join(failures)}"
            )
        return {source.name: record for source, record in zip(plan, results, strict=True)}

    def manifest(self):
        entries = {path.stem: _read_json(path) for path in sorted(self.raw.glob("*.json"))}
        hashes = {
            name: item["sha256"]
            for name, item in entries.items()
            if "sha256" in item and (self.raw / f"{name}.parquet").is_file()
        }
        config_root = Path(__file__).resolve().parents[1] / "config"
        context_file = Path(__file__).with_name("context.json")
        config_files = [*sorted(config_root.glob("*.csv")), context_file]
        config_hashes = {
            path.name: sha256(path.read_bytes()).hexdigest()
            for path in config_files
            if path.is_file()
        }
        identifier = sha256(
            json.dumps([hashes, config_hashes], sort_keys=True).encode()
        ).hexdigest()[:16]
        return {
            "dataset_version": identifier if hashes else None,
            "sources": entries,
            "config": config_hashes,
        }

    def _paths(self, prefix, seasons=None):
        files = sorted(self.raw.glob(f"{prefix}*.parquet"))
        if seasons is None:
            return [str(path) for path in files]
        years = {str(int(year)) for year in seasons}
        return [str(path) for path in files if path.stem.rsplit("_", 1)[-1] in years]

    def _read(self, prefix, seasons=None):
        paths = self._paths(prefix, seasons)
        if not paths:
            return pd.DataFrame()
        with duckdb.connect() as connection:
            return connection.read_parquet(paths, union_by_name=True).df()

    def schedules(self, seasons=None):
        frame = self._read("schedules")
        if frame.empty:
            return frame
        if seasons is not None:
            frame = frame.loc[frame["season"].isin(seasons)]
        return (
            frame.drop_duplicates("game_id")
            .sort_values(["gameday", "game_id"])
            .reset_index(drop=True)
        )

    def injuries(self, seasons=None):
        return self._read("injuries_", seasons).drop_duplicates()

    def rosters(self, seasons=None):
        return self._read("rosters_", seasons).drop_duplicates()

    def depth_charts(self, seasons=None):
        return self._read("depth_", seasons).drop_duplicates()

    def plays(self, seasons=None, charting=True):
        """Return eligible scrimmage decisions ordered by date, game and play."""
        paths = self._paths("pbp_", seasons)
        if not paths:
            return pd.DataFrame()
        with duckdb.connect() as connection:
            pbp = connection.read_parquet(paths, union_by_name=True)
            fields = sorted((PBP_REQUIRED | PBP_OPTIONAL).intersection(pbp.columns))
            eligible = pbp.filter(
                "play_type IN ('run', 'pass') AND down BETWEEN 1 AND 4 "
                "AND posteam IS NOT NULL AND defteam IS NOT NULL AND drive IS NOT NULL "
                "AND ydstogo > 0 AND yardline_100 BETWEEN 1 AND 99"
            )
            first_possession = (
                pbp.filter("posteam IS NOT NULL AND drive IS NOT NULL AND down BETWEEN 1 AND 4")
                .aggregate("game_id, posteam, min(drive) as first_drive", "game_id, posteam")
                .df()
            )
            frame = eligible.project(", ".join(f'"{field}"' for field in fields)).df()
        for field in ("qb_kneel", "qb_spike"):
            if field in frame:
                frame = frame.loc[frame[field].fillna(0).eq(0)]
        frame = frame.drop_duplicates(["game_id", "play_id"]).copy()
        if charting:
            for prefix, key, fields in [
                ("participation_", "play_id", PARTICIPATION),
                ("charting_", "nflverse_play_id", CHARTING),
            ]:
                supplement = self._read(prefix, seasons)
                if not supplement.empty:
                    supplement = supplement.rename(
                        columns={"nflverse_game_id": "game_id", key: "play_id"}
                    )
                    columns = [
                        "game_id",
                        "play_id",
                        *sorted(fields.intersection(supplement.columns)),
                    ]
                    frame = frame.merge(
                        supplement[columns].drop_duplicates(["game_id", "play_id"]),
                        on=["game_id", "play_id"],
                        how="left",
                        validate="one_to_one",
                    )
        frame = frame.rename(columns={"date_pulled": "charting_date_pulled"})
        frame = frame.sort_values(["game_date", "game_id", "play_id"]).reset_index(drop=True)
        frame["call"] = frame["play_type"].where(frame["qb_dropback"].ne(1), "pass")
        frame["is_home"] = (
            frame["posteam_type"].eq("home")
            if "posteam_type" in frame
            else (frame["posteam"].eq(frame["home_team"]) if "home_team" in frame else False)
        )
        frame = frame.merge(
            first_possession, on=["game_id", "posteam"], how="left", validate="many_to_one"
        )
        frame["opening_drive"] = frame["drive"].eq(frame.pop("first_drive"))
        frame["previous_call"] = frame.groupby(["game_id", "posteam", "drive"])["call"].shift()
        schedule = self.schedules(seasons)
        if not schedule.empty:
            context = ["away_rest", "home_rest", "away_qb_id", "home_qb_id", "gametime"]
            columns = ["game_id", *[field for field in context if field in schedule]]
            frame = frame.merge(schedule[columns], on="game_id", how="left", validate="many_to_one")
            if {"away_rest", "home_rest"}.issubset(frame.columns):
                frame["offense_rest"] = frame["home_rest"].where(
                    frame["is_home"], frame["away_rest"]
                )
                frame["defense_rest"] = frame["away_rest"].where(
                    frame["is_home"], frame["home_rest"]
                )
            if {"away_qb_id", "home_qb_id"}.issubset(frame.columns):
                frame["starting_qb_id"] = frame["home_qb_id"].where(
                    frame["is_home"], frame["away_qb_id"]
                )
        if not frame["call"].isin(["run", "pass"]).all():
            raise ValueError("Unexpected playcall label after normalizing play-by-play data")
        return frame

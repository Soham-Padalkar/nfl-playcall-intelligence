"""Optional retrospective neural playcall experiment over NFL opening drives."""

import argparse
import gc
import json
import platform
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from playcall.data import DataStore, _atomic_json
from playcall.models import NUMERIC, SEED, _metrics, chronological_split, load_context

CATEGORICAL = ["posteam", "defteam", "previous_call"]
DRIVE_KEYS = ["game_id", "posteam", "drive"]


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 12
    batch_size: int = 256
    accumulation_steps: int = 1
    hidden_size: int = 48
    history_size: int = 5
    learning_rate: float = 0.001
    seed: int = SEED

    def __post_init__(self):
        if (
            min(
                self.epochs,
                self.batch_size,
                self.accumulation_steps,
                self.hidden_size,
                self.history_size,
            )
            < 1
            or self.learning_rate <= 0
        ):
            raise ValueError("Training sizes, counts and learning rate must be positive.")


def opening_splits(frame, cutoff):
    """Use complete games from strictly ordered seasons and exclude later observations."""
    frame = frame.assign(
        game_date=pd.to_datetime(frame["game_date"], errors="raise").dt.normalize()
    )
    if frame["game_date"].isna().any() or frame["game_id"].isna().any():
        raise ValueError("Games require a valid identifier and date.")
    if frame.groupby("game_id")["game_date"].nunique().gt(1).any():
        raise ValueError("A game appears on multiple dates.")
    frame = frame.loc[frame["game_date"].le(pd.Timestamp(cutoff).normalize())]
    frame = frame.sort_values(["game_date", "game_id", "play_id"])
    frame = frame.loc[frame["opening_drive"]].copy()
    frame["target"] = frame["call"].eq("pass").astype("int64")
    frame["previous_call"] = frame.groupby(DRIVE_KEYS, sort=False)["call"].shift().fillna("none")
    splits = tuple(part.reset_index(drop=True) for part in chronological_split(frame))
    if any(part.empty or part["target"].nunique() < 2 for part in splits):
        raise ValueError("Each split must contain opening-drive runs and passes.")
    for earlier, later in zip(splits[:-1], splits[1:], strict=True):
        if earlier["game_date"].max() >= later["game_date"].min() or not set(
            earlier["game_id"]
        ).isdisjoint(later["game_id"]):
            raise ValueError("Game date or ID overlap between training, validation and test.")
    return splits


def encode_splits(splits, history_size):
    encoder = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler()
                ),
                NUMERIC,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    ).fit(splits[0])
    datasets = []
    for part in splits:
        features = encoder.transform(part).astype(np.float32)
        labels = part["target"].to_numpy(dtype=np.float32)
        position = part.assign(position=np.arange(len(part))).groupby(DRIVE_KEYS, sort=False)[
            "position"
        ]
        history = np.zeros((len(part), history_size, features.shape[1] + 2), dtype=np.float32)
        for column, lag in enumerate(range(history_size, 0, -1)):
            previous = position.shift(lag)
            valid = previous.notna().to_numpy()
            previous_indices = previous.loc[valid].to_numpy(dtype=int)
            history[valid, column, :-2] = features[previous_indices]
            history[valid, column, -2] = labels[previous_indices]
            history[valid, column, -1] = 1
        datasets.append(TensorDataset(*(torch.from_numpy(x) for x in (features, history, labels))))
    return encoder, datasets


class PlaycallNet(nn.Module):
    def __init__(self, kind, input_size, hidden_size):
        super().__init__()
        self.recurrent = (
            nn.GRU(input_size + 2, hidden_size, batch_first=True) if kind == "gru" else None
        )
        self.head = nn.Sequential(
            nn.Linear(input_size + (hidden_size if self.recurrent else 0), hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, current, history):
        if self.recurrent is not None:
            _, hidden = self.recurrent(history)
            current = torch.cat((current, hidden[-1]), dim=1)
        return self.head(current).squeeze(-1)


def _batch_to_device(batch, device):
    return [tensor.to(device, non_blocking=device.type == "cuda") for tensor in batch]


def _loader(data, config, device, shuffle=False):
    return DataLoader(
        data,
        batch_size=config.batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )


@torch.inference_mode()
def _probabilities(model, loader, device):
    model.eval()
    return torch.cat(
        [model(*_batch_to_device(batch[:2], device)).sigmoid().cpu() for batch in loader]
    ).numpy()


def _train_candidate(kind, data, config, device):
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train, validation, test = data
    train_loader = _loader(train, config, device, shuffle=True)
    validation_loader = _loader(validation, config, device)
    model = PlaycallNet(kind, train.tensors[0].shape[1], config.hidden_size).to(device)
    amp = device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp and amp_dtype == torch.float16)
    criterion = nn.BCEWithLogitsLoss(reduction="sum")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    best_loss, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, config.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            current, history, labels = _batch_to_device(batch, device)
            # Average gradients across examples, including a short final optimizer step.
            window = (step // config.accumulation_steps) * config.accumulation_steps
            examples = min(
                config.accumulation_steps * config.batch_size,
                len(train) - window * config.batch_size,
            )
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp):
                loss = criterion(model(current, history), labels) / examples
            scaler.scale(loss).backward()
            if (step + 1) % config.accumulation_steps == 0 or step + 1 == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        val_probs = _probabilities(model, validation_loader, device)
        loss = float(log_loss(validation.tensors[-1].numpy(), val_probs, labels=[0, 1]))
        if loss < best_loss:
            best_loss, best_epoch = loss, epoch
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError("Training did not produce a finite validation loss.")
    model.load_state_dict(best_state)
    test_probs = _probabilities(model, _loader(test, config, device), device)
    return (
        best_state,
        test_probs,
        {
            "best_epoch": best_epoch,
            "validation_log_loss": best_loss,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if amp else platform.processor(),
            "amp_dtype": str(amp_dtype).removeprefix("torch.") if amp else None,
            "peak_allocated_mib": round(torch.cuda.max_memory_allocated(device) / 1048576, 2)
            if amp
            else None,
            "batch_size": config.batch_size,
            "accumulation_steps": config.accumulation_steps,
        },
    )


def _train_with_fallback(kind, data, requested, device):
    config = requested
    fallbacks = []
    while True:
        try:
            state, probabilities, details = _train_candidate(kind, data, config, device)
        except torch.cuda.OutOfMemoryError:
            if device.type != "cuda":
                raise
        else:
            return state, probabilities, {**details, "oom_retries": fallbacks}
        # Drop the failed attempt and retry with its original seed and a smaller batch.
        gc.collect()
        torch.cuda.empty_cache()
        fallbacks.append({"device": str(device), "batch_size": config.batch_size})
        if config.batch_size == 1:
            device, config = torch.device("cpu"), requested
        else:
            smaller = config.batch_size // 2
            steps = (requested.batch_size * requested.accumulation_steps + smaller - 1) // smaller
            config = replace(config, batch_size=smaller, accumulation_steps=steps)


def _checkpoint(directory, kind, state, encoder, config, dataset_version, cutoff):
    directory.mkdir(parents=True, exist_ok=True)
    # Store primitive encoder metadata so weights_only=True can load the artifact.
    numeric = encoder.named_transformers_["numeric"]
    categorical = encoder.named_transformers_["categorical"]
    checkpoint = {
        "model": kind,
        "state_dict": state,
        "config": asdict(config),
        "dataset_version": dataset_version,
        "cutoff": cutoff,
        "features": {
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "numeric_medians": numeric[0].statistics_.tolist(),
            "numeric_means": numeric[1].mean_.tolist(),
            "numeric_scales": numeric[1].scale_.tolist(),
            "categories": [values.tolist() for values in categorical.categories_],
        },
    }
    with NamedTemporaryFile(dir=directory, suffix=".pt", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(directory / f"{kind}.pt")
    finally:
        temporary.unlink(missing_ok=True)


def _evaluate(test, probabilities):
    first_decision = test.groupby(DRIVE_KEYS, sort=False).cumcount().eq(0)
    return {
        "opening_drives": _metrics(test, probabilities),
        "first_eligible_play": _metrics(test.loc[first_decision], probabilities[first_decision]),
    }


def experiment(frame, cutoff, dataset_version, config, device="auto", checkpoints=None):
    """Fit comparison models and score the same frozen holdout for each model."""
    cutoff = pd.Timestamp(cutoff).date().isoformat()
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Select --device cpu or install a CUDA PyTorch build."
        )
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 4))
    splits = opening_splits(frame, cutoff)
    encoder, data = encode_splits(splits, config.history_size)
    train, validation, test = splits
    train_x, _, train_y = data[0].tensors
    val_x, _, val_y = data[1].tensors
    test_x = data[2].tensors[0]
    logistic = LogisticRegression(C=0.35, max_iter=350, random_state=config.seed).fit(
        train_x, train_y
    )
    base_p = float(train_y.mean())
    metrics = {
        "league_mean": _evaluate(test, np.full(len(test), base_p)),
        "logistic": _evaluate(test, logistic.predict_proba(test_x)[:, 1]),
    }
    metrics["league_mean"]["validation_log_loss"] = float(
        log_loss(val_y, np.full(len(validation), base_p))
    )
    metrics["logistic"]["validation_log_loss"] = float(
        log_loss(val_y, logistic.predict_proba(val_x)[:, 1])
    )
    for kind in ("mlp", "gru"):
        state, probabilities, details = _train_with_fallback(kind, data, config, device)
        metrics[kind] = {**_evaluate(test, probabilities), **details}
        if checkpoints is not None:
            _checkpoint(Path(checkpoints), kind, state, encoder, config, dataset_version, cutoff)
    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset_version": dataset_version,
        "experiment_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "cutoff": cutoff,
        "config": asdict(config),
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "python": platform.python_version(),
        "target": "Pass or run among eligible scrimmage decisions on each offense's opening drive",
        "features": {
            "numeric": NUMERIC,
            "categorical": CATEGORICAL,
            "gru_history": "Earlier eligible plays in the same drive, with pre-snap features and observed calls.",
        },
        "splits": {
            name: {
                "first_game_date": str(part["game_date"].min().date()),
                "last_game_date": str(part["game_date"].max().date()),
                "games": int(part["game_id"].nunique()),
                "opening_drives": len(part[DRIVE_KEYS].drop_duplicates()),
                "plays": len(part),
                "seasons": sorted(part["season"].unique().astype(int).tolist()),
            }
            for name, part in zip(("train", "validation", "test"), splits, strict=True)
        },
        "selected_by_validation_log_loss": min(
            metrics, key=lambda kind: metrics[kind]["validation_log_loss"]
        ),
        "metrics": metrics,
        "notes": [
            "Later decisions use observed pre-snap states and call history. First eligible plays measure cold-start performance.",
            "Metrics pool all league test games. They do not estimate the outcome of a particular matchup.",
            "Drive pass-share error compares historical predictions conditional on observed game states with observed pass shares.",
            "The 2026 season is limited to recorded games available at the cutoff. Source records can be revised.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", nargs="+", type=int, default=list(range(2022, 2027)))
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument("--cutoff", default=load_context()["as_of"])
    parser.add_argument("--data", default="data")
    parser.add_argument("--report", default="models/neural-results.json")
    parser.add_argument("--checkpoints", default="models/neural")
    for name, value in asdict(TrainingConfig()).items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=type(value), default=value)
    args = parser.parse_args()
    config = TrainingConfig(**{name: getattr(args, name) for name in asdict(TrainingConfig())})
    store = DataStore(args.data)
    frame = store.plays(args.seasons, charting=False)
    if frame.empty:
        parser.error("Refresh nflverse play-by-play before training.")
    manifest = store.manifest()
    sources = {
        name: source
        for name, source in manifest["sources"].items()
        if name in {f"pbp_{season}" for season in args.seasons}
    }
    report = experiment(
        frame, args.cutoff, manifest["dataset_version"], config, args.device, args.checkpoints
    )
    report["source_files"] = {
        name: {key: source.get(key) for key in ("url", "sha256", "retrieved_at", "rows")}
        for name, source in sources.items()
    }
    destination = Path(args.report)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(destination, report)
    print(
        json.dumps(
            {
                "selected_by_validation_log_loss": report["selected_by_validation_log_loss"],
                "test_log_loss": {
                    name: results["opening_drives"]["log_loss"]
                    for name, results in report["metrics"].items()
                },
                "report": str(destination),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

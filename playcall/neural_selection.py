"""Compare three neural classifiers and export the validation winner for NumPy inference."""

import json
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from playcall.data import DataStore, _atomic_json
from playcall.presnap import (
    CUTOFF,
    HISTORY,
    PERSONNEL,
    VISIBLE_SETUP,
    features,
    play_history,
    scores,
    training_examples,
)

MODEL = Path("models/neural-winner.json")
REPORT = Path("docs/neural-selection.json")
FIELDS = (*VISIBLE_SETUP, *PERSONNEL, *HISTORY)
VISIBLE = (*VISIBLE_SETUP, *PERSONNEL)
NUMERIC = ("down", "ydstogo", "yardline_100", "game_seconds_remaining", "score_differential")
SCALE = np.array([4, 25, 100, 3600, 21], dtype=np.float32)
KINDS = ("MLP", "Residual MLP", "Gated MLP", "GRU")
SEED = 85029


def encode(frame, vocab):
    frame = features(frame)
    numeric = frame.reindex(columns=NUMERIC).fillna(0).to_numpy(dtype=np.float32) / SCALE
    categorical = []
    for field in FIELDS:
        values = frame[field].fillna("").to_numpy(dtype=str)
        scale = 2 if field in PERSONNEL else 1
        categorical.extend(scale * (values == label) for label in vocab[field])
    sequence = frame[HISTORY[-1]].fillna("").str.rjust(3)
    categorical.extend(
        sequence.str[index].eq(token).to_numpy() for index in range(3) for token in ("R", "P")
    )
    return np.column_stack((numeric, *categorical)).astype(np.float32)


def masked(frame):
    return frame.assign(**dict.fromkeys(VISIBLE, pd.NA))


def probabilities(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -35, 35).astype(np.float64)))


class Winner:
    def __init__(self, payload):
        self.name = payload["selected"]
        self.version = payload["version"]
        self.vocab = payload["vocab"]
        self.weights = {
            key: np.array(value, dtype=np.float32) for key, value in payload["weights"].items()
        }

    def predict(self, frame):
        def affine(values, layer):
            return values @ self.weights[layer + ".weight"].T + self.weights[layer + ".bias"]

        inputs = encode(frame, self.vocab)
        if self.name == "GRU":
            sequence = inputs[:, -6:].reshape(-1, 3, 2)
            recurrent = np.zeros((len(inputs), 16), dtype=np.float32)
            for token in sequence.transpose(1, 0, 2):
                ir, iz, inn = np.split(
                    token @ self.weights["history.weight_ih"].T + self.weights["history.bias_ih"],
                    3,
                    axis=1,
                )
                hr, hz, hn = np.split(
                    recurrent @ self.weights["history.weight_hh"].T
                    + self.weights["history.bias_hh"],
                    3,
                    axis=1,
                )
                reset = probabilities(ir + hr).astype(np.float32)
                update = probabilities(iz + hz).astype(np.float32)
                recurrent = np.where(
                    token.any(axis=1, keepdims=True),
                    (1 - update) * np.tanh(inn + reset * hn) + update * recurrent,
                    recurrent,
                )
            inputs = np.column_stack((inputs[:, :-6], recurrent))
        hidden = np.maximum(affine(inputs, "first"), 0)
        activated = np.maximum(affine(hidden, "second"), 0)
        if self.name == "Residual MLP":
            activated = activated + hidden
        elif self.name == "Gated MLP":
            activated = activated * probabilities(affine(hidden, "gate")).astype(np.float32)
        return probabilities(affine(activated, "out")[:, 0])

    def predict_state(self, state):
        state = asdict(state) if is_dataclass(state) else state
        history = play_history(state.get("recent_calls") or (state.get("previous_call"),))
        frame = pd.DataFrame(
            [
                {
                    "down": state.get("down", 1),
                    "ydstogo": state.get("distance", 10),
                    "yardline_100": state.get("yardline_100", 75),
                    "game_seconds_remaining": state.get("game_seconds_remaining", 3600),
                    "score_differential": state.get("score_differential", 0),
                    **history,
                }
            ]
        )
        return float(self.predict(frame)[0])


def load_winner(path=MODEL):
    return Winner(json.loads(Path(path).read_text(encoding="utf-8")))


def fit(store, model_path=MODEL, report_path=REPORT):
    import torch
    from torch import nn

    class Network(nn.Module):
        def __init__(self, kind, width):
            super().__init__()
            self.kind = kind
            self.first = nn.Linear(width + 10 if kind == "GRU" else width, 48)
            self.second = nn.Linear(48, 48)
            self.out = nn.Linear(48, 1)
            self.dropout = nn.Dropout(0.15)
            if kind == "Gated MLP":
                self.gate = nn.Linear(48, 48)
            if kind == "GRU":
                self.history = nn.GRUCell(2, 16)

        def forward(self, x):
            if self.kind == "GRU":
                sequence = x[:, -6:].reshape(-1, 3, 2)
                memory = x.new_zeros(len(x), 16)
                for token in sequence.unbind(dim=1):
                    memory = torch.where(
                        token.any(dim=1, keepdim=True), self.history(token, memory), memory
                    )
                x = torch.cat((x[:, :-6], memory), dim=1)
            h = self.first(x).relu()
            z = self.second(self.dropout(h)).relu()
            if self.kind == "Residual MLP":
                z = h + z
            elif self.kind == "Gated MLP":
                z = z * self.gate(h).sigmoid()
            return self.out(self.dropout(z)).squeeze(1)

    plays = store.plays()
    data = training_examples(plays.loc[pd.to_datetime(plays.game_date).le(pd.Timestamp(CUTOFF))])
    train = data.loc[data.season.lt(2024)]
    val = data.loc[data.season.eq(2024)]
    test = data.loc[data.season.gt(2024)]
    for left, right in ((train, val), (val, test)):
        if (
            left.empty
            or right.empty
            or left.game_date.max() >= right.game_date.min()
            or set(left.game_id) & set(right.game_id)
        ):
            raise ValueError(
                "Training, validation and test must contain separate chronological games."
            )
    vocab = {field: sorted(train[field].dropna().unique().tolist()) for field in FIELDS}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    inputs = torch.as_tensor(encode(train, vocab), device=device)
    blind = torch.as_tensor(encode(masked(train), vocab), device=device)
    y = torch.as_tensor(train.target.to_numpy(dtype=np.float32), device=device)
    val_input = torch.as_tensor(encode(val, vocab), device=device)
    loss_fn = nn.BCEWithLogitsLoss()
    candidates = {}
    for kind in KINDS:
        torch.manual_seed(SEED)
        model = Network(kind, inputs.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
        best = (-1, float("-inf"))
        for epoch in range(1, 31):
            model.train()
            for indexes in torch.randperm(len(train), device=device).split(128):
                # Train on forecasts with and without the current setup.
                use_blind = torch.rand((len(indexes), 1), device=device) < 0.25
                batch = torch.where(use_blind, blind[indexes], inputs[indexes])
                optimizer.zero_grad()
                loss = loss_fn(model(batch), y[indexes])
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_score = scores(
                    val.target.to_numpy(), probabilities(model(val_input).cpu().numpy())
                )
            rank = (val_score["accuracy"], -val_score["log_loss"])
            if rank > best:
                best = rank
                weights = {key: value.cpu().tolist() for key, value in model.state_dict().items()}
                candidates[kind] = (rank, epoch, weights, val_score)
    selected = max(candidates, key=lambda kind: candidates[kind][0])
    metrics = {}
    for kind, (_, epoch, weights, val_score) in candidates.items():
        candidate = Winner({"selected": kind, "version": "", "weights": weights, "vocab": vocab})
        reference = Network(kind, inputs.shape[1]).to(device).eval()
        reference.load_state_dict(
            {key: torch.tensor(value, device=device) for key, value in weights.items()}
        )
        with torch.no_grad():
            expected = probabilities(reference(val_input).cpu().numpy())
        np.testing.assert_allclose(candidate.predict(val), expected, atol=2e-6)
        result = candidate.predict(test)
        atl = test.posteam.eq("ATL").to_numpy()
        first = ~test.duplicated(["game_id", "posteam"]).to_numpy()
        metrics[kind] = {
            "epoch": epoch,
            "validation": val_score,
            "test": scores(test.target.to_numpy(), result),
            "test_atlanta": scores(test.target.to_numpy()[atl], result[atl]),
            "test_atlanta_first_observed": scores(
                test.target.to_numpy()[atl & first], result[atl & first]
            ),
            "test_first_snap_without_visible_setup": scores(
                test.target.to_numpy()[first], candidate.predict(masked(test.loc[first]))
            ),
        }
    payload = {"selected": selected, "vocab": vocab, "weights": candidates[selected][2]}
    payload["version"] = sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
    report = {
        "selected": selected,
        "version": payload["version"],
        "dataset_version": store.manifest()["dataset_version"],
        "cutoff": CUTOFF,
        "device": str(device),
        "seed": SEED,
        "split": {
            name: {"plays": len(part), "games": int(part.game_id.nunique())}
            for name, part in (("train", train), ("validation", val), ("test", test))
        },
        "selection": "Highest 2024 accuracy; lowest validation log loss breaks ties.",
        "candidates": metrics,
    }
    for path, content in ((Path(model_path), payload), (Path(report_path), report)):
        path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(path, content)
    return report


if __name__ == "__main__":
    print(json.dumps(fit(DataStore()), indent=2))

import pandas as pd

from playcall.models import _tactical_distributions


def test_empty_formation_is_unlabeled():
    frame = pd.DataFrame({"offense_formation": ["", " ", "SHOTGUN"], "call": ["pass"] * 3})
    entry = _tactical_distributions(frame)["offense_formation"]
    assert entry["probabilities"] == {"SHOTGUN": 1.0}
    assert entry["coverage"] == 1 / 3

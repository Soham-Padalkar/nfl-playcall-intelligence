import cv2
import numpy as np
import pytest

from playcall.gpu import telemetry
from playcall.video import _overlay, analyze


def test_sample_video(tmp_path):
    path = tmp_path / "fixture.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 144))
    assert out.isOpened()
    for index in range(8):
        out.write(np.full((144, 160, 3), index * 20, np.uint8))
    out.release()
    rows = analyze(path, 1, 3, 2)
    assert [row["frame"] for row in rows] == [1, 3, 5]
    assert rows[-1]["image"].shape == (144, 160, 3)
    with pytest.raises(ValueError):
        analyze(path, 99, 1, 1)


def test_track_continuity():
    img = np.zeros((144, 160, 3), np.uint8)
    _, _, tracks, _, next_id = _overlay(img, [{"box": [10, 10, 30, 50]}], {2: (19, 50)}, 3)
    assert tracks[0]["id"] == 2 and next_id == 3


def test_cpu_fallback(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setattr("shutil.which", lambda _: None)
    info = telemetry()
    assert info["device"] == "cpu" and not info["cuda_available"]

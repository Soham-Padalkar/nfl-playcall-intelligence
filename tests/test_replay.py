import pytest

from playcall.replay import _frozen_evidence


def _branch(call, gain, p_pass, mass, *, turnover=False, children=None):
    prediction = {"p_pass": p_pass} if p_pass is not None else None
    return {
        "call": call,
        "outcome": {"yards": gain, "turnover": turnover},
        "node": {"prediction": prediction, "mass": mass, "children": children or []},
    }


def test_frozen_second_call_marginalizes_only_surviving_mass():
    second = _branch("pass", 8, 0.7, 0.05)
    branches = [
        _branch("run", 4, 0.8, 0.2, children=[second]),
        _branch("pass", 0, 0.2, 0.1),
        _branch("pass", 12, None, 0.2, turnover=True),
    ]
    tree = {"retained_mass": 0.35, "root": {"prediction": {"p_pass": 0.4}, "children": branches}}
    first_play = {"call": "run", "yards_gained": 4}
    second_play = {"call": "pass", "yards_gained": 8}
    result = _frozen_evidence(tree, first_play, second_play)
    assert result["frozen_first_p_pass"] == 0.4
    assert result["frozen_second_conditioning_mass"] == pytest.approx(0.3)
    assert result["frozen_second_p_pass"] == pytest.approx(0.6)
    assert result["two_calls_in_beam"] and result["two_calls_and_outcomes_in_beam"]


def test_turnover_branch_scores_first_only():
    turnover = _branch("pass", 0, None, 0.1, turnover=True)
    tree = {"retained_mass": 0.1, "root": {"prediction": {"p_pass": 0.4}, "children": [turnover]}}
    actual = {"call": "pass", "yards_gained": 0, "interception": 1}
    result = _frozen_evidence(tree, actual)
    assert result["first_call_and_outcome_in_beam"]
    assert result["frozen_second_p_pass"] is None
    assert not result["two_calls_and_outcomes_in_beam"]

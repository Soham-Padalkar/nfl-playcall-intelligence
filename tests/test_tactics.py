import pandas as pd

from playcall.tactics import charted_setups, setup_forecast


def test_tactical_setups_use_joint_examples_and_separate_run_from_pass():
    context = {
        "posteam": "ATL",
        "defteam": "MIN",
        "caller": "Tommy Rees",
        "season": 2025,
        "down": 1,
        "ydstogo": 10,
        "yardline_100": 75,
        "previous_call": None,
    }
    rows = charted_setups(
        pd.DataFrame(
            [
                {
                    **context,
                    "call": "run",
                    "offense_personnel": "12",
                    "qb_location": "U",
                    "run_location": "middle",
                },
                {
                    **context,
                    "call": "pass",
                    "offense_personnel": "11",
                    "qb_location": "S",
                    "is_play_action": True,
                },
                {**context, "call": "pass", "qb_location": "P", "is_screen_pass": True},
            ]
        )
    )
    forecast = setup_forecast(rows, context, {"p_run": 0.4, "p_pass": 0.6})
    assert forecast["pass"]["label_coverage"] == 0.5
    assert forecast["pass"]["options"][0]["personnel"] == "11"
    assert forecast["pass"]["options"][0]["concept"] == "Play-action pass"
    assert forecast["run"]["options"][0]["formation"] == "Under Center"

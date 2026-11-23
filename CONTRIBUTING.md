# Contributing

Read AGENTS.md and the model card. Work on a feature branch and describe the data cutoff, affected population and checks in your pull request. Add a regression test for changes to eligibility, chronology, branch probability mass or snapshot integrity.

Run the following commands in the active Python 3.12 environment.

```powershell
ruff check .
ruff format --check .
python -m pytest -q
python -m pip check
```

Keep source media, raw data, model binaries, tokens and local predictions out of commits. Link to source material with its attribution. Report the denominator and test period with each metric.

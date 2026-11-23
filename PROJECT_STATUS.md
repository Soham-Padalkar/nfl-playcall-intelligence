# Release status

Version 0.1.0 delivers the nine-page Streamlit application, local NFL data cache, tabular and GPU experiments, and timestamped game forecasts. Source and model evaluation are linked below.

| Item | Verified state |
| --- | --- |
| Matchup | Atlanta at Minnesota, November 29, 2026 at noon Central. Zero verified past Tommy Rees matchups against Minnesota under Brian Flores. |
| Sources | 2022-26 play-by-play, schedules, injury and depth records, rosters, FTN charting and available participation files. The 2026 participation file was unavailable. |
| Evaluation | XGBoost achieved 66.93% accuracy across 5,700 held-out opening-drive plays. On Atlanta's first call alone, a separate 27-drive replay favored the league baseline. |
| GPU | RTX 5060 Laptop GPU, driver 573.22, 8,151 MiB, compute capability 12.0. PyTorch 2.7.1+cu128 completed a CUDA matrix product, person detection, and MLP and GRU training. |
| Video | Decoded a public football video. CUDA person detections at frames 30, 75, 120 and 165 numbered 0, 0, 6 and 6. These counts have no ground-truth accuracy labels. |
| Local verification | 21 tests passed with optional PyTorch installed. Ruff lint, formatting, and pip dependency checks passed. All nine pages passed Streamlit AppTest. Desktop and mobile browser widths were checked. |
| Forecast | A real pregame snapshot for a tied game, first and ten at Atlanta's 25, is included with the release. The tree records pruned probability. |
| Updates | The local Windows task refreshes at 06:00 through December 1, 2026. A source refresh and a no-change selective-training check both ran successfully. A fresh clone requires task registration. |
| Revision | Tag v0.1.0 identifies the release commit. Run `git rev-parse HEAD` for the current checkout. |
| Next observation | After kickoff, enter actual plays and elapsed time to compare them with frozen and adaptive predictions. No actual target-game outcome was available at release. |

## Routing verification

Codex CLI 0.167.1 had multi-agent support enabled. Session records verified Astra Medium for architecture and Sol Medium for implementation. Terra Medium inspected the overview screenshot. A GPT-5.6 Luna Low CLI session reviewed supplied source and test text after its separate shell launcher could not inspect the repository. The runtime logged both its model and reasoning setting. Deterministic local checks and GitHub Actions supply the test results. Seven role configurations are included for future work.

## Limits

The direct matchup sample is empty. The full-game approach to scheme comparison uses statistical profiles and available charting. The ensemble assigned similarity zero weight. Field hash and exact tactical labels have limited coverage. The code supports manually reviewed QB alignment, short image-space tracks and optional person detection. There is no validated automatic route or coverage classifier. The optional cloud notebook was checked locally and was not run on a hosted GPU.

[Model card](MODEL_CARD.md) · [Tabular metrics](docs/backtest.json) · [Atlanta replay](docs/replay-results.json) · [GPU metrics](docs/neural-results.json)

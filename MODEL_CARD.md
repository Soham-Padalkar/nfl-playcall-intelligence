# Opening-drive model card

## Selected neural classifier

The current classifier is a gated multilayer perceptron. The comparison trained a plain MLP, a residual MLP, a gated MLP and a GRU on the same 7,256 plays from 2022–23. They used the same pre-snap inputs, including personnel and earlier same-game calls. The GRU processed the last three calls as a sequence. The winner was selected on 3,556 plays from 2024 by accuracy, with log loss as a tie-breaker.

| Architecture | Validation accuracy | Accuracy on 5,700 later plays |
| --- | ---: | ---: |
| MLP | 73.82% | 71.89% |
| Residual MLP | 73.57% | 72.53% |
| Gated MLP | 73.85% | 72.00% |
| GRU | 73.37% | 72.11% |

The residual MLP led on the held-out games. Selecting it on that basis would make those games part of model selection. The gated MLP scored 76.61% on 124 Atlanta plays with archived inputs and 51.01% on the league's 894 first snaps with the visible setup hidden. That test still used the actual initial game state. The current classifier reads only what is available at inference. Simulated future snaps have simulated history and no observed setup. [Neural selection report](docs/neural-selection.json)

## Earlier statistical comparison using the visible setup

The pre-snap predictor uses down, distance, field position, quarterback alignment, offensive personnel, defensive personnel and the last three eligible run or pass calls by the same offense in the same game. It uses each archived call only for later plays. The evaluator considered adding defenders in the box, offensive backfield count and motion. Validation selected the configuration with alignment, personnel and play history. [FTN charting definitions](https://nflreadr.nflverse.com/articles/dictionary_ftn_charting.html)

Personnel groups update the forecast using historical counts in the same situation and alignment. Each matching personnel example has four times the weight of one prior example in that update. Counts with little support still move the estimate by less than counts with many matches. Offensive personnel uses running back and tight end counts, including fullbacks with running backs. Defensive personnel uses defensive line, linebacker and defensive back counts. The 2026 participation file was missing, so the retrospective 2026 rows use alignment and any available prior calls. Live users can enter observed personnel even when the file is missing.

The experiment fit on 2022–23 and selected the input set, shrinkage and confidence threshold on 2024. It evaluated on separate games from September 2025 through November 22, 2026. The comparison model is the previous pre-snap model with alignment, box count, backfield count and motion.

| Held-out result | Previous model | Personnel and history |
| --- | ---: | ---: |
| All eligible calls | 4,061 / 5,700 | 4,070 / 5,700 |
| Calls after an earlier eligible call | 3,436 / 4,806 | 3,441 / 4,806 |
| Atlanta calls | 97 / 124 | 96 / 124 |
| Atlanta first eligible snaps | 21 / 27 | 21 / 27 |

The updated confidence rule requires observed QB alignment and both personnel groups, plus a model probability of at least 90% for the predicted class. It achieved 94.07% accuracy on 489 calls, or 8.58% of the league test plays. A bootstrap that resampled whole games gave a 91.90% lower bound. Coverage depends on seeing the personnel. The frozen pregame forecast uses information available before kickoff. [Pre-snap experiment](docs/presnap-results.json)

The production target is the offensive choice between a run and a dropback on an eligible opening-drive scrimmage play. The population includes both offenses in regular-season and playoff games present in the dataset. The model can support scenario exploration by an analyst. It has no validation for prediction of an exact playbook call or personnel substitution.

## Evaluation

| Split | NFL seasons | Opening-drive decisions | Games |
| --- | --- | ---: | ---: |
| Fit | 2022–23 | 7,256 | 569 |
| Validation | 2024 | 3,556 | 285 |
| Held-out test | 2025–26 through November 22 | 5,700 | 447 |

The team used the validation period for candidate selection, sigmoid calibration and the analogue weight. The selected tabular model is XGBoost. The production fit then uses historical data through the forecast cutoff. The held-out metrics refer to the earlier evaluation fit. Data from earlier games can include later publisher corrections.

| Candidate | Test log loss | Test Brier |
| --- | ---: | ---: |
| League frequency | 0.6852 | 0.2460 |
| Situational frequency | 0.6004 | 0.2100 |
| Three-component matchup ensemble | 0.6013 | 0.2102 |
| Random forest | 0.5997 | 0.2082 |
| XGBoost | **0.5838** | **0.2023** |

XGBoost correctly classified 66.93% of the eligible plays. On 124 Atlanta test plays it achieved 66.13% accuracy, 0.5736 log loss and a 0.2004 Brier score. These Atlanta observations span different playcallers. The test measures later decisions after the analyst knows the actual prior state. It does not measure joint accuracy for a complete frozen drive tree. Top-two accuracy adds no information for two classes.

## Matchup evidence

A uses verified caller intervals against Flores in Minnesota. The current matchup has zero previous qualifying games. B compares defensive profiles and shrinks a caller-specific or team-level rate toward situational evidence. C uses recent offense and defense decisions. D combines A, B and C with sample-size-aware weights. Validation gave B zero weight. XGBoost won model selection independently of D. The dashboard exposes D for comparison.

In a separate experiment with common training games, Model B had 0.60254 log loss using play-by-play profiles and 0.60317 with added charting. Feature permutation in the selected tabular model increased log loss most for down (0.1833) and distance (0.1070). Permutation measures predictive dependence and does not establish causation.

## Interpretation

- Tactical probabilities use observed historical labels where available and show their support. Snap depth, routes, coverages and targets can be incomplete. No validated detector recovers the full play from broadcast footage.
- The reported interval describes the empirical pass rate in the reference sample. It is not a confidence interval for an individual model score.
- The simulation enumerates possible plays and empirical outcomes. An exact-yardage beam can discard most path mass by depth three. Use the reported retained mass to interpret the displayed branches.
- Injuries, projected starters and inactive lists are context. Their effects have no validated predictive coefficients.
- No target-game accuracy exists before kickoff. The postgame evaluation requires the actual sequence and its saved pregame or adaptive forecast.

Named target and carrier shares use the offense’s latest five opening drives in its most recent season and require five labeled examples. They describe usage conditional on a recorded name. Injuries and substitutions can change who is available.

The optional GPU MLP scored 0.6270 log loss and the GRU scored 0.6296 on the same 5,700 eligible decisions. Neither surpassed the production tabular model. The [neural experiment](docs/neural-results.json) also reports first-play scores.

## Frozen Atlanta replay

A separate XGBoost fit through 2024 used identity calibration and the actual initial field position, down, score and clock. It generated two decisions before seeing any outcomes. Across 27 Atlanta opening drives, it predicted the first call correctly on 15 drives (55.6%). The position-specific league baseline predicted 18 correctly (66.7%). First-call log loss was 0.7015 for XGBoost and 0.6853 for the baseline.

On 26 drives with a second eligible play, the frozen second-call estimate had 0.7051 log loss. The league baseline had 0.7049. The width-eight beam retained the observed two-call sequence in 16 drives and both calls with their exact gains and turnover status in three. Average retained mass after two plays was 11.25%. These samples give no evidence of a matchup-specific advantage on Atlanta's first two calls. [Reproduce the replay](docs/replay-results.json)

# Design decisions

| Decision | Reason and tradeoff |
| --- | --- |
| Start with run/pass conditional on an eligible scrimmage play. | This target is measurable in play-by-play data. Exact concepts and assignment responsibilities require independently reviewed labels. Fourth-down kicks and administrative plays require separate treatment. |
| Store local Parquet and query with DuckDB. | The tabular data fit a single machine. A hosted database would add setup costs without improving this initial workflow. |
| Verify actual caller intervals and coordinator tenure. | Coach titles or franchise identity can misattribute games. A caller can face Minnesota while coaching a different team. |
| Use whole-game chronological evaluation. | Random play splits can leak information from the same game or later coaching periods. Source revisions remain a separate retrospective limitation. |
| Shrink estimates and report sample sizes. | A few opening drives can produce extreme raw rates. Evidence for ensemble weights must precede the evaluation period. |
| Track discarded tree mass and forecast origin. | A bounded tree is practical for interaction. Retaining only the largest branches without recording their discarded mass would misstate confidence. An immutable initial forecast enables honest comparison with live updates. |
| Keep video optional and local. | Media rights and camera framing vary. Broadcast coordinates, occluded players and partial formations place limits on inference. |
| Use native Codex roles and a manual CLI handoff. | The installed CLI exposes multi-agent support. A second orchestration system would add operational cost. Role files set model and reasoning; the run record verifies execution. |

The custom-agent file format follows the [Codex subagent documentation](https://developers.openai.com/codex/subagents/).

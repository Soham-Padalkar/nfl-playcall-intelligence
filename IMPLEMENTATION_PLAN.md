# Implementation plan

Each row describes an acceptance gate. A working page or a prototype alone does not close a research question.

| Milestone | Dependencies and work | Acceptance evidence |
| --- | --- | --- |
| 1. Functional MVP | Inspect Git, hosting, Python and GPU. Ingest play-by-play, validate keys and expose probabilities in Streamlit. | Reproducible refresh on real data, known row and game counts, successful dashboard smoke run, normalized baseline and hardware or CPU execution evidence. |
| 2. Matchup research | Use M1 data. Verify actual playcaller by date, Vikings coordinator tenure and schedule. Profile similar defenses and recent Atlanta plays. | Primary-source links and coverage dates, direct game list including other teams, subset tests, feature definitions, sample counts and explicit gaps. |
| 3. Models | Use M2 filters. Train candidates and baselines, fit ensemble weights using earlier data, evaluate on later games. | Game-disjoint chronological split, Brier score, log loss, accuracy, calibration, opening-drive sample size and baseline comparison. Record version, seed and limitations. |
| 4. Opening-drive forecast | Use M3 probabilities and empirical outcomes. Branch through football states and record frozen and adaptive forecasts. | State-transition tests, normalized conditional probabilities, terminal and discarded mass, immutable snapshot test and post-play comparison. |
| 5. Video analysis | Use a permitted local video and M1 hardware support. Sample frames, detect and track players, review uncertain annotations. | Playback, overlay and tracking smoke checks, explicit scope of any formation inference, CPU fallback and measured CUDA test when available. |
| 6. Evaluation and release | Integrate M1–M5, exercise error states, test chronological integrity, publish code and measured results. | Passing tests and CI, checked public files, documented limitations, user-facing launch commands, real commit, tag and public repository link. |

Optional advanced candidates enter the dashboard when labels, sample size and chronological evaluations support their use. Log absent inputs and defer unsupported tactical outputs.

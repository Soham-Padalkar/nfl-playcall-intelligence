# Sources and availability

The refresh uses public releases from the nflverse project. Each local sidecar records its source URL, retrieval time, content hash, row count and availability notes. Downloads can change when the upstream project corrects a game. Historical replays use present-day revisions.

| Source | Use |
| --- | --- |
| [nflverse play-by-play](https://github.com/nflverse/nflverse-data/releases/tag/pbp) | Eligible offensive decisions, pre-snap state, observed outcomes and prior-game summaries |
| [nflverse schedules](https://github.com/nflverse/nfldata/blob/master/data/games.csv) | Dates, opponents, location and scheduling context |
| [Participation](https://github.com/nflverse/nflverse-data/releases/tag/pbp_participation) | Observed personnel, formation and defensive labels when published |
| [FTN charting](https://github.com/nflverse/nflverse-data/releases/tag/ftn_charting) | Retrospective play charting. Credit FTN Data via nflverse |
| [Injuries](https://github.com/nflverse/nflverse-data/releases/tag/injuries) and [depth charts](https://github.com/nflverse/nflverse-data/releases/tag/depth_charts) | Dated personnel context. Missing reports have unknown status |
| [Atlanta schedule](https://www.atlantafalcons.com/schedule/2026/) | Atlanta at Minnesota, November 29, noon Central at U.S. Bank Stadium |
| [Falcons playcaller announcement](https://www.atlantafalcons.com/news/tommy-rees-offensive-play-caller-kevin-stefanski) and [Browns handoff](https://www.clevelandbrowns.com/news/offensive-coordinator-tommy-rees-to-call-plays-for-the-browns-team-coverage) | Tommy Rees called Atlanta’s plays in 2026 and Cleveland’s from 2025 Week 10 |
| [Minnesota coaching biography](https://www.vikings.com/team/coaches-roster/brian-flores) | Brian Flores has coordinated Minnesota’s defense since 2023 |

The MIT license covers this project's code. The data, pretrained weights, team publications and footage retain their own terms. Consult the [nflverse data repository](https://github.com/nflverse/nflverse-data) and the individual source release before redistribution. The public repository contains no game footage or raw datasets.

The video smoke check used AFN Naples footage by U.S. Navy Petty Officer 2nd Class Chance Hanson. The public-domain source is documented on [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:AFN_Naples_TV_Spot_-_Army_vs_Navy_Game_(1028703).webm). This clip checks decoding and inference. It supplies no tactical training labels.

Player roster changes are checked against [nflverse annual rosters](https://github.com/nflverse/nflverse-data/releases/tag/rosters).

# Architecture

The Python package is playcall/. Streamlit loads the application from app.py. The local machine performs ingestion and conventional modeling on the CPU. Video analysis and neural experiments can select CUDA after a device check.

## Data contract

The ingestion boundary accepts season identifiers and a local data directory. It produces validated play-level Parquet data and source metadata. DuckDB reads Parquet for analysis. Identify each play by game_id and play_id. Preserve game date, offense, defense, drive, down, distance, field position, clock and score where available. Keep unavailable values null. A published retrieval date establishes when this installation obtained the data. Historical availability of specific field revisions requires separate evidence.

Playcaller records need a team, date interval, responsibility and a source. Direct matchups require the actual offensive caller in that interval and Minnesota defending during Flores' coordinator tenure. Keep evidence about a coaching title separate from evidence about who called plays. The schedule determines whether the target game is upcoming or historical.

## Prediction contract

Given a pre-snap state and an as-of date, the modeling layer estimates two probabilities conditional on an eligible offensive run or pass. Tactical attributes require their own observed labels and support counts. Use only prior games for matchup, similarity, recent tendency and league estimates. Apply shrinkage to limited samples. Show the sample and the component weights with the output. Evaluate feature ablations and candidate models across game-date cutoffs. A historical replay can still be affected by subsequent source corrections.

The simulation layer branches over decisions and empirical outcomes. State includes down, distance and yards to the opponent's goal line. The sum of retained, terminated and pruned branch probabilities must account for the incoming mass. Freeze the initial tree with a timestamp, model version, dataset version, Git revision and inputs. Record observed plays and adaptive predictions separately so evaluation can recover what each forecast knew.

## Presentation and optional services

The dashboard presents matchup evidence, tree exploration, playcaller and defensive comparisons, historical tendencies, video analysis, model evaluation, live entry and system telemetry. Heavy processing belongs behind explicit actions or offline commands. The video service accepts media the operator may analyze. Keep uploads and derived frames local. Label image coordinates, detections and tactical inferences according to what the footage supports. GPU verification requires a successful tensor computation on the device.

The application can function with missing optional evidence. Show the date, coverage and reason for an unavailable input. PROJECT_STATUS.md records verified implementation coverage against this design.

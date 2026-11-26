"""Video review, model evaluation, and hardware telemetry."""

import json
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import plotly.express as px
import streamlit as st

from playcall import ui


@st.cache_data(max_entries=3, show_spinner="Sampling video frames…")
def _sample_video(content, suffix, start, count, stride, neural, batch):
    from playcall.video import analyze

    with NamedTemporaryFile(suffix=suffix, delete=False) as stream:
        stream.write(content)
        local = Path(stream.name)
    try:
        return analyze(
            local, start_frame=start, count=count, stride=stride, neural=neural, batch_size=batch
        )
    finally:
        local.unlink(missing_ok=True)


def _snap_alignment(depth, uncertainty, back_alignment):
    """Classify only the measured depth interval and operator-reviewed back location."""
    if depth is None or uncertainty is None:
        return "Unknown · no depth or measurement uncertainty"
    low, high = max(0, depth - uncertainty), depth + uncertainty
    if high <= 1:
        return "Under-center depth"
    if low >= 2.5 and high <= 4.5 and back_alignment == "Directly behind":
        return "Pistol-like alignment"
    if low >= 4.5 and back_alignment in {"Offset", "Empty"}:
        return "Shotgun-like alignment"
    return "Ambiguous alignment"


def video():
    from cv2 import error as OpenCVError

    st.caption(
        "Inspect an authorized local clip. People, line candidates, and motion are shown in image coordinates."
    )
    uploaded = st.file_uploader("Video file", type=["mp4", "mov", "avi", "webm", "mkv"])
    if uploaded is None:
        return
    content = uploaded.getvalue()
    digest = sha256(content).hexdigest()
    suffix = Path(uploaded.name).suffix.lower()
    media_type = {
        "mp4": "video/mp4",
        "mov": "video/quicktime",
        "avi": "video/x-msvideo",
        "webm": "video/webm",
        "mkv": "video/x-matroska",
    }
    st.video(content, format=media_type[suffix.lstrip(".")])
    st.caption("For broad browser playback support, use an H.264 MP4 clip.")
    with st.form("sample-video"):
        cols = st.columns(3)
        start = cols[0].number_input("First frame (zero-based)", min_value=0, value=0)
        count = cols[1].number_input("Frames to sample", 1, 60, 4)
        stride = cols[2].number_input("Frame interval", 1, 600, 5)
        cols = st.columns(2)
        detector = cols[0].selectbox(
            "Person detector", ["OpenCV HOG (CPU)", "MobileNet Faster R-CNN"]
        )
        batch = cols[1].number_input("Inference batch size", 1, 16, 2)
        analyze = st.form_submit_button("Analyze frames", type="primary")
    if analyze:
        try:
            rows = _sample_video(
                content,
                suffix,
                start,
                count,
                stride,
                detector == "MobileNet Faster R-CNN",
                batch,
            )
        except (ValueError, RuntimeError, OSError, ImportError, OpenCVError) as error:
            st.error(f"Frame analysis failed. {error}")
        else:
            st.session_state["video_result"] = (digest, rows)
    stored = st.session_state.get("video_result")
    if stored is None or stored[0] != digest:
        return
    frames = {row["frame"]: row for row in stored[1]}
    frame = st.selectbox("Reviewed frame", list(frames), key=f"frame-{digest}")
    row = frames[frame]
    st.image(row["image"], width="stretch")
    st.caption(
        f"{row['method']} · Green people boxes · Blue line candidates · Orange track displacement"
    )
    cols = st.columns(3)
    cols[0].metric("People detected", len(row["detections"]))
    cols[1].metric("Sampled tracks", len(row["tracks"]))
    cols[2].metric("Line candidates", len(row["lines"]))
    with st.expander("Positions and detector scores"):
        if row["detections"]:
            st.dataframe(row["detections"], hide_index=True)
            st.dataframe(row["tracks"], hide_index=True)
        st.caption(
            "Detector scores are uncalibrated. Camera movement contributes to image displacement."
        )
    st.subheader("Formation and motion review")
    st.caption(
        "Use a manual label when the frame shows enough of the offense to verify a formation or pre-snap motion."
    )
    with st.form("frame-review", clear_on_submit=True):
        formation = st.text_input("Formation")
        motion = st.selectbox("Pre-snap motion", ["Unreviewed", "Observed", "None visible"])
        st.caption(
            "Optional snap-depth check. Enter field-calibrated yards only when you can identify the QB, center, and field marks."
        )
        measures = st.columns(3)
        depth = measures[0].number_input("QB depth behind center (yd)", 0.0, 25.0, None, step=0.25)
        uncertainty = measures[1].number_input(
            "Measurement uncertainty (± yd)", 0.0, 25.0, None, step=0.25
        )
        back_alignment = measures[2].selectbox(
            "Back alignment", ["Unknown", "Directly behind", "Offset", "Empty"]
        )
        notes = st.text_area("Review notes")
        annotation = st.form_submit_button("Save frame annotation")
    if annotation:
        labels = st.session_state.setdefault("video_annotations", {}).setdefault(digest, {})
        labels[frame] = {
            "frame": frame,
            "formation": formation.strip(),
            "motion": motion,
            "depth_yd": depth,
            "measurement_uncertainty_yd": uncertainty,
            "back_alignment": back_alignment,
            "experimental_snap_alignment": _snap_alignment(depth, uncertainty, back_alignment),
            "notes": notes.strip(),
        }
        st.success(f"Saved the annotation for frame {frame}.")
    labels = st.session_state.get("video_annotations", {}).get(digest, {})
    if labels:
        st.dataframe([labels[key] for key in sorted(labels)], width="stretch", hide_index=True)
        st.caption(
            "Snap alignment is an experimental rule based on manually reviewed depth and back position. "
            "The measurement range is operator-supplied. "
            "[Nevada pistol formation reference](https://nevadawolfpack.com/news/2010/9/30/The_Pistol_Packs_a_Punch_Around_College_Football)"
        )
        st.download_button(
            "Export annotations",
            json.dumps(list(labels.values()), indent=2),
            file_name=f"{Path(uploaded.name).stem}-annotations.json",
            mime="application/json",
        )


def _neural_results():
    locations = (
        ui.REPORT.with_name("neural-results.json"),
        ui.ROOT / "docs" / "neural-results.json",
    )
    path = next((path for path in locations if path.is_file()), None)
    if path is None:
        return
    results = json.loads(path.read_text(encoding="utf-8"))
    best = results["selected_by_validation_log_loss"]
    metrics = results["metrics"]
    names = {
        "gru": "PyTorch GRU",
        "mlp": "PyTorch MLP",
        "league_mean": "League frequency",
        "logistic": "Logistic regression",
    }
    with st.expander("Neural experiment and baselines"):
        st.caption(
            f"Separate chronological experiment · Dataset {results['dataset_version']} · "
            f"Validation selected {names[best]}"
        )
        cohorts = {
            "All opening-drive plays": "opening_drives",
            "First eligible play": "first_eligible_play",
        }
        cohort = st.selectbox("Neural test sample", list(cohorts))
        rows = []
        for key, result in metrics.items():
            scored = result[cohorts[cohort]]
            rows.append(
                {
                    "Model": names[key],
                    "Test plays": scored["n"],
                    "Accuracy": scored["accuracy"],
                    "Log loss": scored["log_loss"],
                    "Brier score": scored["brier"],
                    "Device used": result.get("device", "cpu").upper(),
                    "Peak GPU allocation (MiB)": result.get("peak_allocated_mib"),
                }
            )
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
            column_config={
                "Accuracy": st.column_config.NumberColumn("Accuracy", format="percent"),
            },
        )
        st.caption(
            "These candidates were evaluated separately from the dashboard’s deployed prediction model."
        )
        st.download_button(
            "Download neural experiment",
            json.dumps(results, indent=2),
            file_name="neural-results.json",
            mime="application/json",
        )


def laboratory():
    neural_report = ui.ROOT / "docs" / "neural-selection.json"
    if neural_report.is_file():
        comparison = json.loads(neural_report.read_text(encoding="utf-8"))
        st.subheader("Neural classifier comparison")
        st.metric("Validation winner", comparison["selected"])
        st.dataframe(
            [
                {
                    "Architecture": name,
                    "Validation accuracy": item["validation"]["accuracy"],
                    "Held-out accuracy": item["test"]["accuracy"],
                    "Held-out Atlanta accuracy": item["test_atlanta"]["accuracy"],
                    "First snap without setup": item["test_first_snap_without_visible_setup"][
                        "accuracy"
                    ],
                }
                for name, item in comparison["candidates"].items()
            ],
            hide_index=True,
            column_config={
                label: st.column_config.NumberColumn(label, format="percent")
                for label in (
                    "Validation accuracy",
                    "Held-out accuracy",
                    "Held-out Atlanta accuracy",
                    "First snap without setup",
                )
            },
        )
    model = ui.get_model()
    if model:
        report = model.report
    elif ui.REPORT.is_file():
        report = json.loads(ui.REPORT.read_text(encoding="utf-8"))
    else:
        st.info("Train the local models to produce a chronological backtest report.")
        _neural_results()
        return
    st.subheader("Archived tabular backtest")
    selected = report["selected_model"]
    metrics = report["metrics"]
    st.caption(
        f"Model {report['model_version']} · Training data through {report['training_cutoff']}"
    )
    cols = st.columns(4)
    cols[0].metric("Selected model", selected)
    cols[1].metric("Test accuracy", f"{metrics[selected]['accuracy']:.1%}")
    cols[2].metric("Test log loss", f"{metrics[selected]['log_loss']:.4f}")
    cols[3].metric("Test Brier score", f"{metrics[selected]['brier']:.4f}")
    st.subheader("Chronological test results")
    rows = [
        {
            "Model": name,
            "Plays": values["n"],
            "Accuracy": values["accuracy"],
            "Log loss": values["log_loss"],
            "Brier score": values["brier"],
            "Calibration error": values["calibration_error"],
            "Drive pass-share MAE": values["drive_pass_share_mae"],
        }
        for name, values in metrics.items()
    ]
    st.dataframe(
        rows,
        width="stretch",
        hide_index=True,
        column_config={"Accuracy": st.column_config.NumberColumn("Accuracy", format="percent")},
    )
    with st.expander("Date and game coverage"):
        st.dataframe(pd.DataFrame.from_dict(report["splits"], orient="index"), width="stretch")
    st.subheader("Reliability by model")
    apparent_models = list(metrics)
    name = st.selectbox("Backtest model", apparent_models, index=apparent_models.index(selected))
    scopes = ["League test sample"]
    if name == selected and report.get("atlanta_selected_metrics"):
        scopes.append("Atlanta test sample")
    scope = st.selectbox("Test sample", scopes)
    evaluation = (
        report["atlanta_selected_metrics"] if scope == "Atlanta test sample" else metrics[name]
    )
    reliability = [
        {"forecast": item["predicted_pass"], "observed": item["observed_pass"], "n": item["n"]}
        for item in evaluation["reliability"]
    ]
    ui.reliability(reliability, name)
    st.caption(
        f"{evaluation['n']:,} plays · Accuracy {evaluation['accuracy']:.1%} · "
        f"Log loss {evaluation['log_loss']:.4f} · Brier {evaluation['brier']:.4f}"
    )
    importance = report.get("feature_importance")
    if importance:
        st.subheader("Feature importance")
        st.caption(
            "Change in validation log loss after shuffling each feature. Larger increases indicate more predictive value in this comparison."
        )
        features = pd.DataFrame(importance).sort_values("log_loss_increase").tail(12)
        fig = px.bar(
            features,
            x="log_loss_increase",
            y="feature",
            orientation="h",
            color_discrete_sequence=[ui.PASS],
            labels={"log_loss_increase": "Increase in log loss", "feature": "Feature"},
        )
        ui.figure(fig, height=max(310, len(features) * 25 + 95))
    if model and not getattr(model, "neural_winner", None):
        from playcall.simulation import GameState

        st.subheader("Opening-state ensemble contributions")
        prediction = model.predict(GameState())
        ui.show_components(prediction)
    st.caption(
        f"Defensive analog weight selected on validation data · {report['ensemble_analog_weight']:.2f}"
    )
    with st.expander("Backtest methodology"):
        for note in report["notes"]:
            st.write(note)
        st.download_button(
            "Download full backtest",
            json.dumps(report, indent=2),
            file_name="backtest.json",
            mime="application/json",
        )
    _neural_results()


def monitor():
    from playcall import gpu

    st.caption("Local readings from NVIDIA and PyTorch.")
    info = gpu.telemetry()
    ready = ui.MODEL.is_file()
    cols = st.columns(3)
    cols[0].metric("NVIDIA GPU", info.get("gpu_name", "Unavailable"))
    cols[1].metric("GPU memory", f"{info['vram_mib']} MiB" if "vram_mib" in info else "Unavailable")
    cols[2].metric("PyTorch CUDA", "Available" if info["cuda_available"] else "Unavailable")
    if "used_mib" in info and "vram_mib" in info:
        try:
            total = float(info["vram_mib"])
            used = float(info["used_mib"])
            if total > 0:
                st.progress(
                    max(0.0, min(1.0, used / total)), text=f"{used:,.0f} of {total:,.0f} MiB in use"
                )
        except (TypeError, ValueError):
            st.caption("The NVIDIA driver did not report a numeric memory reading.")
    cols = st.columns(3)
    cols[0].metric("PyTorch version", info.get("torch_version", "Unavailable"))
    cols[1].metric("Tabular model device", "CPU" if ready else "Awaiting training")
    cols[2].metric("Tabular model", "Checkpoint present" if ready else "Unavailable")
    if st.button("Run PyTorch tensor check", type="primary"):
        st.session_state["tensor_check"] = gpu.verify()
    if result := st.session_state.get("tensor_check"):
        if result.get("verified"):
            st.success(f"Tensor operation passed on {result['device'].upper()}.")
        else:
            st.error("PyTorch could not verify the tensor operation.")
        st.json(result, expanded=False)
    with st.expander("Driver and process telemetry"):
        st.json(info)
    st.subheader("Dataset refresh status")
    ui.source_table(ui.get_manifest())

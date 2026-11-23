"""Sample user-supplied video. All coordinates are image pixels."""

import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=1)
def _hog():
    detector = cv2.HOGDescriptor()
    detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    return detector


@lru_cache(maxsize=2)
def _network(device):
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    return (
        fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        )
        .to(device)
        .eval()
    )


def _detect_net(frames, device):
    import torch

    with (
        torch.inference_mode(),
        torch.autocast(device_type=device, enabled=device == "cuda"),
    ):
        inputs = [
            torch.from_numpy(frame).permute(2, 0, 1).to(device).float() / 255 for frame in frames
        ]
        outputs = _network(device)(inputs)
        detections = []
        for output in outputs:
            keep = (output["labels"] == 1) & (output["scores"] >= 0.55)
            boxes = output["boxes"][keep].cpu().int().tolist()
            scores = output["scores"][keep].cpu().tolist()
            detections.append(
                [{"box": box, "score": float(score)} for box, score in zip(boxes, scores)]
            )
        return detections


def _detect(frames, neural):
    if neural:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            return _detect_net(frames, device), "MobileNet Faster R-CNN / " + device
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return _detect_net(frames, "cpu"), "MobileNet Faster R-CNN / CPU fallback"
    result = []
    for frame in frames:
        if frame.shape[0] < 128 or frame.shape[1] < 64:
            result.append([])
            continue
        boxes, scores = _hog().detectMultiScale(frame, winStride=(8, 8), scale=1.08)
        result.append(
            [
                {"box": [int(x), int(y), int(x + w), int(y + h)], "score": float(score)}
                for (x, y, w, h), score in zip(boxes, scores)
            ]
        )
    return result, "OpenCV HOG / CPU"


def _overlay(rgb, detections, previous, next_id):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        70,
        minLineLength=max(50, rgb.shape[1] // 4),
        maxLineGap=12,
    )
    candidates = [] if lines is None else lines[:, 0].tolist()[:20]
    image = rgb.copy()
    for x1, y1, x2, y2 in candidates:
        cv2.line(image, (x1, y1), (x2, y2), (40, 150, 255), 1)
    points = {}
    tracks = []
    remaining = previous.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        point = ((x1 + x2) // 2, y2)
        nearest = min(remaining, key=lambda key: math.dist(point, remaining[key]), default=None)
        displacement = None
        if nearest is not None and math.dist(point, remaining[nearest]) < max(24, (y2 - y1) * 0.75):
            track_id = nearest
            prior = remaining.pop(nearest)
            displacement = round(math.dist(point, prior), 1)
            cv2.line(image, prior, point, (255, 180, 70), 2)
        else:
            track_id = next_id
            next_id += 1
        points[track_id] = point
        tracks.append(
            {
                "id": track_id,
                "x": point[0],
                "y": point[1],
                "displacement_px": displacement,
            }
        )
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 235, 145), 2)
        cv2.putText(
            image,
            f"#{track_id}",
            (x1, max(14, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 235, 145),
            1,
        )
    return image, candidates, tracks, points, next_id


def analyze(path, start_frame=0, count=4, stride=5, neural=False, batch_size=2):
    if start_frame < 0 or not 1 <= count <= 60 or stride < 1 or not 1 <= batch_size <= 16:
        raise ValueError("Invalid frame range or batch size.")
    cap = cv2.VideoCapture(str(Path(path)))
    if not cap.isOpened():
        raise ValueError("OpenCV could not read this video.")
    rows, previous, next_id = [], {}, 1
    try:
        for offset in range(0, count, batch_size):
            frames, indices = [], []
            for i in range(offset, min(count, offset + batch_size)):
                index = start_frame + i * stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                if w > 960:
                    frame = cv2.resize(frame, (960, max(1, round(960 * h / w))))
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                indices.append(index)
            if not frames:
                break
            detected, method = _detect(frames, neural)
            for index, frame, boxes in zip(indices, frames, detected):
                image, lines, tracks, previous, next_id = _overlay(frame, boxes, previous, next_id)
                rows.append(
                    {
                        "frame": index,
                        "image": image,
                        "detections": boxes,
                        "lines": lines,
                        "tracks": tracks,
                        "method": method,
                    }
                )
    finally:
        cap.release()
    if not rows:
        raise ValueError("No frames could be decoded in the selected range.")
    return rows

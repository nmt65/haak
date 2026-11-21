"""Live YOLO26n detection on frames sent by the dashboard.

Uses the same detector and the same confidence threshold (0.25) as the WiMANS
camera features in nv_benchmark/vision.py, so the camera and fusion models see
live inputs built the same way as their training inputs.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np

from server.explain import decode_data_url

YOLO_CONFIDENCE = 0.25  # same as nv_benchmark/vision.py
YOLO_IMAGE_SIZE = 320  # small input keeps CPU latency around 35 ms
LEFT_SECTOR_END = 1 / 3
RIGHT_SECTOR_START = 2 / 3


def sector_of(center_x: float) -> str:
    """Same thirds as the dashboard and as vision.py's left/center/right person counts."""
    if center_x < LEFT_SECTOR_END:
        return "left"
    if center_x >= RIGHT_SECTOR_START:
        return "right"
    return "center"


def detections_from_boxes(xyxy: np.ndarray, confidences: np.ndarray, labels: list[str], width: int, height: int) -> list[dict]:
    """Converts pixel boxes into the dashboard's detection format (coordinates 0–1)."""
    detections = []
    for (x1, y1, x2, y2), confidence, label in zip(xyxy, confidences, labels):
        box = {"x": float(x1 / width), "y": float(y1 / height), "width": float((x2 - x1) / width), "height": float((y2 - y1) / height)}
        center_x = box["x"] + box["width"] / 2
        detections.append({
            "label": label,
            "confidence": round(float(confidence), 3),
            "bbox": box,
            "horizontalSector": sector_of(center_x),
            "relativeDepth": None,
        })
    return detections


class YoloLiveDetector:
    """Loads YOLO26n on first use and runs one frame at a time (frames arriving while busy are skipped)."""

    def __init__(self, weights: Path):
        self.weights = weights
        self.model = None
        self.status = "neîncărcat"
        self.error: str | None = None
        self.last_latency_ms: float | None = None
        self.lock = threading.Lock()

    def is_busy(self) -> bool:
        return self.lock.locked()

    def detect(self, frame_data_url: str) -> list[dict]:
        with self.lock:
            try:
                if self.model is None:
                    if not self.weights.is_file():
                        raise FileNotFoundError(f"Lipsesc greutățile {self.weights.name}")
                    from ultralytics import YOLO

                    self.status = "se încarcă"
                    self.model = YOLO(str(self.weights))
                    self.status = "gata"
                # RGB -> BGR (OpenCV channel order, as in training); contiguous copy for the model.
                image = np.ascontiguousarray(np.asarray(decode_data_url(frame_data_url))[:, :, ::-1])
                height, width = image.shape[:2]
                start = time.perf_counter()
                result = self.model.predict(image, verbose=False, conf=YOLO_CONFIDENCE, imgsz=YOLO_IMAGE_SIZE)[0]
                self.last_latency_ms = (time.perf_counter() - start) * 1000
                boxes = result.boxes
                labels = [result.names[int(class_id)] for class_id in boxes.cls.tolist()]
                self.error = None
                return detections_from_boxes(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy(), labels, width, height)
            except Exception as exc:  # surface the real error in the dashboard
                self.status = "eroare"
                self.error = f"{type(exc).__name__}: {exc}"
                raise

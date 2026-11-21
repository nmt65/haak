"""Loads the checkpoints written by train.py and runs them on live inputs.

Each checkpoint stores its own normalisation (fitted on training data only) and
the decision threshold chosen on validation, so live predictions use exactly the
same preprocessing as the benchmark.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from nv_benchmark.models import PerceptionClassifier

VISUAL_FEATURE_COUNT = 8


@dataclass
class LoadedModel:
    mode: str  # "csi" | "camera" | "fusion"
    model: PerceptionClassifier
    threshold: float
    csi_mu: np.ndarray
    csi_sd: np.ndarray
    visual_mu: np.ndarray
    visual_sd: np.ndarray


@dataclass
class Prediction:
    probability: float
    is_present: bool
    latency_ms: float


def load_checkpoint(path: Path) -> LoadedModel:
    # weights_only=False: our own checkpoints also store numpy normalisation arrays.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["task"] != "binary":
        raise ValueError(f"{path.name}: live pipeline expects the binary presence task")
    model = PerceptionClassifier(
        checkpoint["mode"],
        channels=checkpoint["channel_count"],
        visual_dim=checkpoint["visual_dim"],
        n_outputs=1,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return LoadedModel(
        mode=checkpoint["mode"],
        model=model,
        threshold=float(checkpoint["threshold"]),
        csi_mu=checkpoint["csi_mu"],
        csi_sd=checkpoint["csi_sd"],
        visual_mu=checkpoint["visual_mu"],
        visual_sd=checkpoint["visual_sd"],
    )


class ModelRuntime:
    """Holds the CSI / camera / fusion models from one benchmark run directory."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.models: dict[str, LoadedModel] = {}
        self.is_synthetic = True
        self.dataset = "necunoscut"
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.is_synthetic = bool(metrics.get("synthetic_fixture", True))
            self.dataset = str(metrics.get("dataset", "necunoscut"))
        for mode in ("csi", "camera", "fusion"):
            checkpoint = run_dir / f"{mode}.pt"
            if checkpoint.is_file():
                self.models[mode] = load_checkpoint(checkpoint)

    def describe(self) -> dict:
        return {
            "run_dir": str(self.run_dir),
            "dataset": self.dataset,
            "trained_on_synthetic_data": self.is_synthetic,
            "available_models": sorted(self.models),
        }

    def predict(self, mode: str, csi_features: np.ndarray | None, visual_features: np.ndarray | None) -> Prediction | None:
        """Runs one model on one sample. Returns None if that model or its inputs are missing."""
        loaded = self.models.get(mode)
        if loaded is None:
            return None
        needs_csi = mode in ("csi", "fusion")
        needs_camera = mode in ("camera", "fusion")
        if (needs_csi and csi_features is None) or (needs_camera and visual_features is None):
            return None

        csi = np.zeros((1, 12, 128), dtype=np.float32) if csi_features is None else csi_features[None]
        csi = np.clip((csi - loaded.csi_mu) / loaded.csi_sd, -10, 10).astype(np.float32)
        visual = np.zeros((1, VISUAL_FEATURE_COUNT), dtype=np.float32) if visual_features is None else visual_features[None]
        visual = np.clip((visual - loaded.visual_mu) / loaded.visual_sd, -10, 10).astype(np.float32)

        start = time.perf_counter()
        with torch.no_grad():
            logit = loaded.model(torch.from_numpy(csi), torch.from_numpy(visual)).item()
        latency_ms = (time.perf_counter() - start) * 1000
        probability = float(1 / (1 + np.exp(-np.clip(logit, -30, 30))))
        return Prediction(probability=probability, is_present=probability >= loaded.threshold, latency_ms=latency_ms)


def person_features_from_detections(detections: list[dict]) -> np.ndarray:
    """Builds the 8 camera features used in training (see nv_benchmark/vision.py) from ONE frame.

    Order: mean people, max people, mean max confidence, left/center/right person counts,
    mean person box area, observed fraction. With a single frame, mean == max.
    The live detections come from the browser's YOLOE, while WiMANS training used YOLO26n,
    so the camera and fusion models face a (small) detector domain shift here.
    """
    people = [d for d in detections if d.get("label") == "person"]
    count = float(len(people))
    max_confidence = max((float(d["confidence"]) for d in people), default=0.0)
    sectors = [d.get("horizontalSector") for d in people]
    areas = [float(d["bbox"]["width"]) * float(d["bbox"]["height"]) for d in people]
    return np.array(
        [
            count,
            count,
            max_confidence,
            float(sectors.count("left")),
            float(sectors.count("center")),
            float(sectors.count("right")),
            float(np.mean(areas)) if areas else 0.0,
            1.0,
        ],
        dtype=np.float32,
    )

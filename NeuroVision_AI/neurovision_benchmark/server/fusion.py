"""Builds the structured world state from camera detections and the CSI window.

The world state only contains what the sensors support:
  * objects and their direction come from the camera;
  * radio presence / motion comes from the CSI model and says NOTHING about position;
  * sectors without camera detections are "unknown", never "free".
"""
from __future__ import annotations

import time

import numpy as np

from server.csi_sources import CsiWindow, motion_energy
from server.model_runtime import ModelRuntime, person_features_from_detections

SECTORS = ("left", "center", "right")
# Camera and radio observations further apart than this are not fused (temporal synchronisation).
MAX_SYNC_GAP_S = 1.0
CAMERA_STALE_AFTER_S = 2.0
# Motion energy above this (normalised feature units) is reported as "movement in the channel".
MOTION_THRESHOLD = 0.06
CSI_PLOT_STEPS = 64  # downsampled time steps sent to the dashboard for the signal plot


def _prediction_to_dict(prediction) -> dict | None:
    if prediction is None:
        return None
    return {
        "probability": round(prediction.probability, 3),
        "is_present": bool(prediction.is_present),
        "latency_ms": round(prediction.latency_ms, 3),
    }


def _agreement(camera_sees_person: bool | None, radio_says_present: bool | None) -> str:
    """How the two modalities relate. Disagreement is reported, not hidden."""
    if camera_sees_person is None and radio_says_present is None:
        return "no_data"
    if camera_sees_person is None:
        return "radio_only"
    if radio_says_present is None:
        return "camera_only"
    if camera_sees_person and radio_says_present:
        return "agree_present"
    if not camera_sees_person and not radio_says_present:
        return "agree_absent"
    return "camera_person_radio_absent" if camera_sees_person else "radio_present_camera_none"


def _downsample_for_plot(features: np.ndarray) -> list[list[float]]:
    """Picks evenly spaced time steps so the plot payload stays small."""
    steps = np.linspace(0, features.shape[1] - 1, CSI_PLOT_STEPS).astype(int)
    return np.round(features[:, steps], 3).tolist()


def build_world_state(
    runtime: ModelRuntime,
    csi_window: CsiWindow | None,
    csi_error: str | None,
    camera_message: dict | None,
) -> dict:
    now = time.time()

    detections: list[dict] = []
    camera_age_s = None
    if camera_message is not None:
        camera_age_s = now - camera_message["received_at"]
        if camera_age_s <= CAMERA_STALE_AFTER_S:
            detections = camera_message.get("detections", [])
    is_camera_fresh = camera_age_s is not None and camera_age_s <= CAMERA_STALE_AFTER_S

    # Fuse only observations that are close in time.
    is_synchronised = (
        csi_window is not None and is_camera_fresh and abs(csi_window.captured_at - camera_message["received_at"]) <= MAX_SYNC_GAP_S
    )

    csi_features = csi_window.features if csi_window is not None else None
    visual_features = person_features_from_detections(detections) if is_camera_fresh else None

    csi_prediction = runtime.predict("csi", csi_features, None)
    camera_prediction = runtime.predict("camera", None, visual_features)
    fusion_prediction = runtime.predict("fusion", csi_features, visual_features) if is_synchronised else None

    camera_sees_person = any(d.get("label") == "person" for d in detections) if is_camera_fresh else None
    radio_says_present = csi_prediction.is_present if csi_prediction is not None else None
    motion = motion_energy(csi_features) if csi_features is not None else None

    observed_sectors = {d.get("horizontalSector") for d in detections}
    unknown_sectors = [sector for sector in SECTORS if sector not in observed_sectors]

    return {
        "type": "world_state",
        "timestamp": now,
        "objects": [
            {
                "label": d.get("label"),
                "confidence": d.get("confidence"),
                "sector": d.get("horizontalSector"),
                "bbox": d.get("bbox"),
                "relative_depth": d.get("relativeDepth"),
                "source": "camera",
            }
            for d in detections
        ],
        "unknown_sectors": unknown_sectors,
        "camera": {
            "is_fresh": is_camera_fresh,
            "age_s": None if camera_age_s is None else round(camera_age_s, 2),
            "person_count": int(visual_features[0]) if visual_features is not None else None,
            # "camera" = real detections from the browser (YOLOE) or from YOLO26n on this server.
            "source": None if camera_message is None else camera_message.get("source", "camera"),
            "detector": "nicio cameră" if camera_message is None else camera_message.get("detector", "YOLOE în browser"),
        },
        "radio": None
        if csi_window is None
        else {
            "mode": csi_window.mode,
            "is_simulated": csi_window.is_simulated,
            "description": csi_window.description,
            "sample_id": csi_window.sample_id,
            "ground_truth_presence": csi_window.ground_truth_presence,
            "motion_energy": None if motion is None else round(motion, 4),
            "motion_detected": None if motion is None else motion > MOTION_THRESHOLD,
            "plot": _downsample_for_plot(csi_window.features),
            "position_information": "niciuna: CSI nu indică poziția sau identitatea",
        },
        "radio_error": csi_error,
        "models": {
            "csi": _prediction_to_dict(csi_prediction),
            "camera": _prediction_to_dict(camera_prediction),
            "fusion": _prediction_to_dict(fusion_prediction),
        },
        "fusion": {
            "is_synchronised": is_synchronised,
            "agreement": _agreement(camera_sees_person, radio_says_present),
        },
        "runtime": runtime.describe(),
    }

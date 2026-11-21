"""Live YOLO26n path: box conversion, feature parity with training, and a real run when weights exist."""
import base64
import io
from pathlib import Path

import numpy as np
import pytest

from server.live_detector import YoloLiveDetector, detections_from_boxes, sector_of
from server.model_runtime import person_features_from_detections

WEIGHTS = Path(__file__).resolve().parent.parent / "yolo26n.pt"


def test_sector_thirds_match_training_features():
    assert sector_of(0.1) == "left"
    assert sector_of(0.5) == "center"
    assert sector_of(0.9) == "right"


def test_boxes_are_normalised_and_features_match_vision_py():
    xyxy = np.array([[0, 0, 100, 200], [500, 100, 600, 300]], dtype=np.float32)
    detections = detections_from_boxes(xyxy, np.array([0.9, 0.6]), ["person", "person"], width=640, height=480)
    assert detections[0]["bbox"] == pytest.approx({"x": 0.0, "y": 0.0, "width": 100 / 640, "height": 200 / 480})

    # vision.py's per-frame row: count, max conf, left/center/right counts, mean area.
    centers = (xyxy[:, 0] + xyxy[:, 2]) / (2 * 640)
    areas = ((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])) / (640 * 480)
    expected = [2, 2, 0.9, (centers < 1 / 3).sum(), ((centers >= 1 / 3) & (centers < 2 / 3)).sum(), (centers >= 2 / 3).sum(), areas.mean(), 1.0]
    assert person_features_from_detections(detections) == pytest.approx(expected, rel=1e-5)


@pytest.mark.skipif(not WEIGHTS.is_file(), reason="yolo26n.pt not downloaded")
def test_yolo26n_finds_people_in_sample_image():
    from PIL import Image
    from ultralytics.utils import ASSETS

    buffer = io.BytesIO()
    Image.open(ASSETS / "bus.jpg").convert("RGB").save(buffer, format="JPEG")
    data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()

    detections = YoloLiveDetector(WEIGHTS).detect(data_url)
    labels = [d["label"] for d in detections]
    assert labels.count("person") >= 3
    assert all(0 <= d["bbox"]["x"] <= 1 for d in detections)

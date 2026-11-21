"""Tests for the live backend: CSI sources, fusion honesty rules and explanations."""
import numpy as np

from server.csi_sources import SimulatedCsiSource, motion_energy, parse_esp32_csi_line
from server.explain import grounded_evidence, rule_based_explanation
from server.fusion import _agreement
from server.model_runtime import person_features_from_detections


def test_simulated_csi_is_always_labelled_simulated():
    window = SimulatedCsiSource().read_window()
    assert window.features.shape == (12, 128)
    assert window.is_simulated is True
    assert "simulat" in window.description


def test_simulated_presence_adds_motion():
    source = SimulatedCsiSource(seed=1)
    source.started_at -= 0  # first period: nobody present
    empty = motion_energy(source.read_window().features)
    source.started_at -= 8  # shift into the "person present" period
    occupied = motion_energy(source.read_window().features)
    assert occupied > empty


def test_esp32_line_parser():
    line = "CSI_DATA,STA,AA:BB:CC:DD:EE:FF,-40,11,1,0,0,0,0,0,0,0,0,-90,0,6,0,0,0,0,0,8,[3 4 0 5 -6 8 1 0]"
    amplitudes = parse_esp32_csi_line(line)
    assert np.allclose(amplitudes, [5, 5, 10, 1])
    assert parse_esp32_csi_line("I (123) wifi: connected") is None


def test_person_features_from_browser_detections():
    detections = [
        {"label": "person", "confidence": 0.9, "horizontalSector": "left", "bbox": {"width": 0.2, "height": 0.5}},
        {"label": "chair", "confidence": 0.8, "horizontalSector": "right", "bbox": {"width": 0.1, "height": 0.1}},
    ]
    features = person_features_from_detections(detections)
    assert features.tolist() == [1.0, 1.0, np.float32(0.9), 1.0, 0.0, 0.0, np.float32(0.1), 1.0]


def test_disagreement_is_reported():
    assert _agreement(True, False) == "camera_person_radio_absent"
    assert _agreement(False, True) == "radio_present_camera_none"
    assert _agreement(None, True) == "radio_only"


def _world(objects, csi_present, simulated=True):
    return {
        "camera": {"is_fresh": True},
        "objects": objects,
        "radio": {"mode": "simulated", "is_simulated": simulated, "motion_detected": True},
        "models": {"csi": {"probability": 0.8 if csi_present else 0.2, "is_present": csi_present}},
        "fusion": {"agreement": "radio_present_camera_none" if csi_present and not objects else "agree_present"},
        "unknown_sectors": ["right"],
    }


def test_rule_based_explanation_never_locates_radio_presence():
    text = rule_based_explanation(_world([], csi_present=True))
    assert "nu permit identificarea sursei sau a poziției" in text
    assert "simulate" in text
    assert "necunoscute, nu libere" in text


def test_rule_based_explanation_names_camera_objects_by_side():
    objects = [{"label": "person", "sector": "left", "confidence": 0.9}]
    assert "persoană în stânga" in rule_based_explanation(_world(objects, csi_present=True))


def test_grounded_evidence_has_no_radio_position():
    evidence = grounded_evidence(_world([], csi_present=True))
    assert evidence["radio_position_information"] == "niciuna"
    assert evidence["camera_objects"] == []

"""Wi-Fi CSI input sources for the live pipeline.

Three modes, each one honest about what it is:
  * "simulated" - generated signal, ALWAYS labelled as simulated, never real radio data.
  * "recorded"  - replays samples of a prepared dataset file (.npz from prepare_wimans.py).
  * "hardware"  - reads an ESP32 running ESP32-CSI-Tool over a serial port.
                  Ordinary laptops and routers do NOT expose CSI; this needs dedicated hardware.

Every source returns a CsiWindow: 12 feature channels x 128 time steps, the same
representation the trained models use (see nv_benchmark/features.py).
"""
from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from nv_benchmark.features import csi_to_features

SIMULATED_SAMPLE_RATE_HZ = 100
WINDOW_SECONDS = 3.0  # WiMANS recordings are 3 s long, so live windows use the same length
SIMULATED_PRESENCE_PERIOD_S = 8.0  # simulated "person" appears for 8 s, then leaves for 8 s
RECORDED_SECONDS_PER_SAMPLE = 3.0


@dataclass
class CsiWindow:
    features: np.ndarray  # float32 [12, 128]
    mode: str
    is_simulated: bool
    captured_at: float  # unix time, seconds
    description: str
    ground_truth_presence: int | None = None  # only known for recorded / simulated data
    sample_id: str | None = None


class SimulatedCsiSource:
    """Synthetic CSI amplitude (time x antennas x subcarriers). For demos and tests ONLY."""

    mode = "simulated"

    def __init__(self, seed: int = 7):
        self.random = np.random.default_rng(seed)
        self.started_at = time.time()

    def is_person_present(self, now: float) -> bool:
        elapsed = now - self.started_at
        return int(elapsed / SIMULATED_PRESENCE_PERIOD_S) % 2 == 1

    def read_window(self) -> CsiWindow:
        now = time.time()
        steps = int(WINDOW_SECONDS * SIMULATED_SAMPLE_RATE_HZ)
        signal = 2.0 + 0.3 * self.random.standard_normal((steps, 3, 30))
        is_present = self.is_person_present(now)
        if is_present:
            # A moving body perturbs the channel: add a slow oscillation plus extra variance.
            t = np.linspace(0, WINDOW_SECONDS, steps)
            signal += (0.7 * np.sin(t * 12 + now))[:, None, None]
            signal += 0.25 * self.random.standard_normal((steps, 3, 30))
        return CsiWindow(
            features=csi_to_features(signal.astype(np.float32)),
            mode=self.mode,
            is_simulated=True,
            captured_at=now,
            description="CSI simulat: semnal generat, nu măsurători radio reale.",
            ground_truth_presence=int(is_present),
        )


class RecordedCsiSource:
    """Replays prepared dataset samples (one 3-second recording every 3 seconds)."""

    mode = "recorded"

    def __init__(self, npz_path: Path, held_out_ids: set[str] | None = None):
        with np.load(npz_path, allow_pickle=False) as data:
            self.features = data["x"].astype(np.float32)
            self.labels = data["y"].astype(np.int64)
            self.ids = data["ids"].astype(str)
            self.dataset_name = str(data["dataset"].item())
            self.is_synthetic_file = bool(data["synthetic"].item())
        # Prefer replaying held-out test recordings, which the model never trained on.
        indices = np.arange(len(self.ids))
        if held_out_ids:
            held_out = np.array([sample_id in held_out_ids for sample_id in self.ids])
            if held_out.any():
                indices = indices[held_out]
        self.order = indices
        self.position = 0
        self.last_switch = 0.0

    def read_window(self) -> CsiWindow:
        now = time.time()
        if now - self.last_switch >= RECORDED_SECONDS_PER_SAMPLE:
            self.position = (self.position + 1) % len(self.order)
            self.last_switch = now
        index = int(self.order[self.position])
        origin = "fixture sintetic" if self.is_synthetic_file else f"set de date {self.dataset_name}"
        return CsiWindow(
            features=self.features[index],
            mode=self.mode,
            is_simulated=self.is_synthetic_file,
            captured_at=now,
            description=f"CSI înregistrat ({origin}), proba {self.ids[index]}. Nu este o măsurătoare în timp real.",
            ground_truth_presence=int(self.labels[index]) if len(np.unique(self.labels)) == 2 else None,
            sample_id=str(self.ids[index]),
        )


# ESP32-CSI-Tool prints lines like: CSI_DATA,STA,<mac>,<rssi>,...,<len>,[i0 r0 i1 r1 ...]
_CSI_LINE = re.compile(r"CSI_DATA.*\[(?P<values>[-\d\s]+)\]")


def parse_esp32_csi_line(line: str) -> np.ndarray | None:
    """Turns one ESP32-CSI-Tool line into subcarrier amplitudes, or None if it is not a CSI line."""
    match = _CSI_LINE.search(line)
    if not match:
        return None
    raw_values = [int(value) for value in match.group("values").split()]
    if len(raw_values) < 4 or len(raw_values) % 2:
        return None
    pairs = np.asarray(raw_values, dtype=np.float32).reshape(-1, 2)  # (imaginary, real) per subcarrier
    return np.hypot(pairs[:, 0], pairs[:, 1])


@dataclass
class HardwareCsiSource:
    """Reads ESP32-CSI-Tool output from a serial port in a background thread.

    NOT validated with physical hardware in this project: treat its output as
    unverified until tested with your own device, band and antenna setup.
    """

    port: str
    baud_rate: int = 921600
    mode: str = "hardware"
    buffer: deque = field(default_factory=lambda: deque(maxlen=2000))
    error: str | None = None

    def __post_init__(self):
        try:
            import serial  # pyserial, optional dependency
        except ImportError as exc:
            raise RuntimeError("Modul hardware necesită pachetul pyserial (pip install pyserial).") from exc
        self.connection = serial.Serial(self.port, self.baud_rate, timeout=1)
        self.running = True
        threading.Thread(target=self._read_forever, daemon=True).start()

    def _read_forever(self):
        while self.running:
            try:
                line = self.connection.readline().decode("ascii", errors="ignore")
            except Exception as exc:  # serial errors must reach the dashboard, not vanish
                self.error = f"Eroare port serial: {exc}"
                self.running = False
                return
            amplitudes = parse_esp32_csi_line(line)
            if amplitudes is not None:
                self.buffer.append((time.time(), amplitudes))

    def read_window(self) -> CsiWindow:
        if self.error:
            raise RuntimeError(self.error)
        now = time.time()
        recent = [amplitudes for captured, amplitudes in self.buffer if now - captured <= WINDOW_SECONDS]
        if len(recent) < 10:
            raise RuntimeError("Nu sosesc pachete CSI de la placă. Verifică portul și firmware-ul ESP32-CSI-Tool.")
        width = min(len(a) for a in recent)
        signal = np.stack([a[:width] for a in recent])  # time x subcarriers
        return CsiWindow(
            features=csi_to_features(signal),
            mode=self.mode,
            is_simulated=False,
            captured_at=now,
            description=f"CSI real de la {self.port}: {len(recent)} pachete în ultimele {WINDOW_SECONDS:.0f} s. Neverificat științific.",
        )

    def close(self):
        self.running = False
        self.connection.close()


def motion_energy(features: np.ndarray) -> float:
    """Average absolute change over time (feature channel 10). Higher = more movement in the channel."""
    mean_absolute_delta = features[10]
    return float(np.mean(mean_absolute_delta))

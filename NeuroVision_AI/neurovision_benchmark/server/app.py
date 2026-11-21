"""NeuroVision AI live backend (FastAPI).

    camera detections (from the browser) ─┐
                                          ├─► models ─► fusion ─► world state ─► WebSocket ─► dashboard
    Wi-Fi CSI (simulated/recorded/hardware)┘

Start:  python -m uvicorn server.app:app --port 8000   (from the neurovision_benchmark folder)
"""
from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from server.csi_sources import HardwareCsiSource, RecordedCsiSource, SimulatedCsiSource
from server.explain import QwenExplainer, rule_based_explanation
from server.fusion import build_world_state
from server.live_detector import YoloLiveDetector
from server.model_runtime import ModelRuntime

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_DIR / "runs"
WORLD_STATE_INTERVAL_S = 0.2  # 5 updates per second to the dashboard

# Real-data runs are preferred; the synthetic smoke run is only a clearly labelled fallback.
RUN_DIR_PREFERENCE = ["wimans_fusion", "wimans_csi", "synthetic_smoke"]
RECORDED_DATA_PREFERENCE = ["wimans_paired.npz", "wimans_presence.npz", "synthetic_fixture.npz"]
SMOKE_RUN_NAME = "synthetic_smoke"
SMOKE_TRAINING_EPOCHS = 3


def _first_existing(base: Path, names: list[str], required_file: str | None = None) -> Path | None:
    for name in names:
        candidate = base / name
        if candidate.exists() and (required_file is None or (candidate / required_file).exists()):
            return candidate
    return None


def _held_out_ids(run_dir: Path) -> set[str] | None:
    split_file = run_dir / "heldout_indices.npz"
    if not split_file.is_file():
        return None
    with np.load(split_file, allow_pickle=False) as split:
        return set(split["test_ids"].astype(str).tolist())


def _create_synthetic_smoke_run() -> Path:
    """First start of a fresh checkout (runs/ is not in git): train the small, clearly
    labelled SYNTHETIC run so the backend can start. Its numbers are software checks only."""
    print("Nu există niciun model antrenat: se antrenează rularea de test pe date SINTETICE (câteva secunde).", flush=True)
    fixture = PROJECT_DIR / "data" / "processed" / "synthetic_fixture.npz"
    if not fixture.exists():
        subprocess.run([sys.executable, "make_synthetic_fixture.py"], cwd=PROJECT_DIR, check=True)
    run_dir = RUNS_DIR / SMOKE_RUN_NAME
    subprocess.run(
        [sys.executable, "train.py", "--data", str(fixture), "--mode", "all", "--epochs", str(SMOKE_TRAINING_EPOCHS), "--run-dir", str(run_dir)],
        cwd=PROJECT_DIR,
        check=True,
    )
    return run_dir


class LiveState:
    """Everything the live pipeline needs, shared by the REST routes and the WebSocket."""

    def __init__(self):
        configured_run = os.environ.get("NV_RUN_DIR")
        run_dir = Path(configured_run) if configured_run else _first_existing(RUNS_DIR, RUN_DIR_PREFERENCE, "metrics.json")
        if run_dir is None:
            run_dir = _create_synthetic_smoke_run()
        self.runtime = ModelRuntime(run_dir)
        self.recorded_npz = _first_existing(PROJECT_DIR / "data" / "processed", RECORDED_DATA_PREFERENCE)
        self.csi_source = SimulatedCsiSource()
        self.csi_error: str | None = None
        self.camera_message: dict | None = None
        self.last_world: dict | None = None
        self.qwen = QwenExplainer()
        self.yolo = YoloLiveDetector(PROJECT_DIR / "yolo26n.pt")

    def detect_frame(self, message: dict):
        """Runs YOLO26n on a frame from the dashboard and makes its detections the camera input."""
        detections = self.yolo.detect(message["frame"])
        self.camera_message = {
            "type": "camera",
            "source": message.get("source", "camera"),
            "detections": detections,
            "received_at": time.time(),
            "detector": "YOLO26n (backend)",
        }

    def set_csi_mode(self, mode: str, port: str | None = None):
        old_source = self.csi_source
        if mode == "simulated":
            self.csi_source = SimulatedCsiSource()
        elif mode == "recorded":
            if self.recorded_npz is None:
                raise ValueError("Nu există date înregistrate pregătite în data/processed/.")
            self.csi_source = RecordedCsiSource(self.recorded_npz, _held_out_ids(self.runtime.run_dir))
        elif mode == "hardware":
            if not port:
                raise ValueError("Modul hardware cere portul serial al plăcii (de ex. COM3).")
            self.csi_source = HardwareCsiSource(port)
        else:
            raise ValueError(f"Mod CSI necunoscut: {mode}")
        if isinstance(old_source, HardwareCsiSource):
            old_source.close()
        self.csi_error = None

    def compute_world(self) -> dict:
        try:
            window = self.csi_source.read_window()
            self.csi_error = None
        except Exception as exc:  # e.g. hardware not sending packets: show it, keep running
            window = None
            self.csi_error = str(exc)
        self.last_world = build_world_state(self.runtime, window, self.csi_error, self.camera_message)
        return self.last_world


state = LiveState()
app = FastAPI(title="NeuroVision AI backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:4173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class CsiModeRequest(BaseModel):
    mode: str
    port: str | None = None


class ExplainRequest(BaseModel):
    frame: str | None = None  # JPEG data URL captured in the browser
    use_qwen: bool = False


@app.get("/api/status")
def get_status():
    return {
        "runtime": state.runtime.describe(),
        "csi_mode": state.csi_source.mode,
        "csi_error": state.csi_error,
        "recorded_data": None if state.recorded_npz is None else state.recorded_npz.name,
        "qwen_status": state.qwen.status,
        "qwen_error": state.qwen.error,
        "yolo_status": state.yolo.status,
        "yolo_error": state.yolo.error,
        "yolo_weights_present": state.yolo.weights.is_file(),
        "yolo_latency_ms": None if state.yolo.last_latency_ms is None else round(state.yolo.last_latency_ms, 1),
        "camera_connected": state.camera_message is not None,
    }


@app.post("/api/csi")
def set_csi_mode(request: CsiModeRequest):
    try:
        state.set_csi_mode(request.mode, request.port)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"csi_mode": state.csi_source.mode}


def _summarise_run(metrics_path: Path) -> dict:
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {
        "run": metrics_path.parent.relative_to(RUNS_DIR).as_posix(),
        "dataset": metrics["dataset"],
        "synthetic": metrics["synthetic_fixture"],
        "seed": metrics["seed"],
        "test_samples": metrics["cohort"]["test"],
        "models": {
            mode: {**result["test"], "latency_p50_ms": result["test_latency"]["p50_ms"]}
            for mode, result in metrics["models"].items()
        },
    }


@app.get("/api/benchmarks")
def get_benchmarks():
    """All measured results found under runs/. Synthetic runs are flagged, never mixed with real ones."""
    if not RUNS_DIR.is_dir():
        return {"runs": [], "aggregates": []}
    runs = [_summarise_run(path) for path in sorted(RUNS_DIR.rglob("metrics.json"))]
    aggregates = []
    for path in sorted(RUNS_DIR.rglob("aggregate.json")):
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        aggregate["run"] = path.parent.relative_to(RUNS_DIR).as_posix()
        aggregates.append(aggregate)
    return {"runs": runs, "aggregates": aggregates}


@app.post("/api/explain")
async def explain(request: ExplainRequest):
    world = state.last_world or state.compute_world()
    response = {"rule_based": rule_based_explanation(world), "qwen": None, "qwen_error": None}
    if request.use_qwen:
        if not request.frame:
            response["qwen_error"] = "Qwen are nevoie de un cadru de la cameră."
        else:
            try:
                response["qwen"] = await run_in_threadpool(state.qwen.explain, request.frame, world)
            except Exception as exc:
                response["qwen_error"] = f"Qwen nu a putut genera descrierea: {exc}"
    return response


@app.websocket("/ws/live")
async def live(websocket: WebSocket):
    await websocket.accept()

    async def run_yolo(message: dict):
        try:
            await run_in_threadpool(state.detect_frame, message)
        except Exception:
            pass  # the error is kept in state.yolo.error and shown via /api/status

    async def receive_camera_messages():
        while True:
            message = json.loads(await websocket.receive_text())
            if message.get("type") == "camera":
                # Detections computed in the browser (YOLOE-26n-seg).
                message["received_at"] = time.time()
                message["detector"] = "YOLOE în browser"
                state.camera_message = message
            elif message.get("type") == "frame" and not state.yolo.is_busy():
                # A JPEG frame for YOLO26n; frames arriving while YOLO is busy are skipped.
                asyncio.create_task(run_yolo(message))

    receiver = asyncio.create_task(receive_camera_messages())
    try:
        while not receiver.done():
            world = await run_in_threadpool(state.compute_world)
            await websocket.send_text(json.dumps(world))
            await asyncio.sleep(WORLD_STATE_INTERVAL_S)
    except WebSocketDisconnect:
        pass
    finally:
        receiver.cancel()

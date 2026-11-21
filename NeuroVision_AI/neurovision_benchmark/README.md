# NeuroVision AI — Reproducible Sensor Benchmark

**Status:** Tested reference benchmark scaffold; not a clinically validated navigation aid, not a neural stimulation system. It does not build a brain implant. It DOES NOT include or redistribute the public datasets or pretrained model weights.

## State of this copy (21 Nov 2026)

| Item | Status |
|---|---|
| Scaffold tests (`pytest`) | 15 pass (4 original + 11 for the live backend, including a real YOLO26n run) |
| Synthetic smoke benchmark, 3 seeds | runs; results in `runs/synthetic_three_seed/` are **software checks only** |
| WiMANS benchmark | **not run yet**: needs your Kaggle token; then `.\train_wimans.ps1 -WithYolo` |
| YOLO26n | weights in `yolo26n.pt` (5.5 MB, official Ultralytics release); ~35 ms per 320 px frame on this CPU; used for WiMANS video features and, optionally, live in the backend |
| MM-Fi action model | not run: needs MM-Fi access + the official toolbox (section 2) |
| Live FastAPI backend + dashboard integration | working with simulated / recorded CSI |
| Qwen3-VL-2B explanations | working on CPU: ~17 s first load, ~40 s per explanation |
| Real CSI hardware (ESP32-CSI-Tool) | parser + serial reader implemented, **not tested with a board** |

Changes to the original scaffold: early stopping (`--patience`, default 3), false positive rate, `metrics.csv` / `aggregate.csv`, precision/recall/FPR/latency in seed aggregates, `pyyaml` added to requirements.

## Live backend (FastAPI) — connects to the NeuroVision dashboard

```powershell
# one-time setup (a short venv path avoids the Windows 260-character path limit)
python -m venv $env:LOCALAPPDATA\neurovision\venv
& $env:LOCALAPPDATA\neurovision\venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-server.txt
# start (from this folder)
& $env:LOCALAPPDATA\neurovision\venv\Scripts\python.exe -m uvicorn server.app:app --port 8000
```

Then in the dashboard: **Senzori și fuziune → Conectează backend-ul**.

- `server/csi_sources.py` — CSI modes: `simulated` (always labelled), `recorded` (replays held-out samples of a prepared `.npz`), `hardware` (ESP32-CSI-Tool over serial; ordinary laptops/routers do not expose CSI).
- `server/model_runtime.py` — loads `csi.pt` / `camera.pt` / `fusion.pt` with their training-only normalisation and validation threshold. Prefers `runs/wimans_fusion`, then `runs/wimans_csi`, then the labelled synthetic run. Override with `NV_RUN_DIR`.
- `server/fusion.py` — world state: camera objects + direction, radio presence/motion **without position**, agreement/disagreement, unknown sectors, 1 s temporal synchronisation.
- `server/explain.py` — deterministic Romanian explanation (always) + optional Qwen3-VL-2B with only the structured evidence. Qwen text is unverified and never drives alerts.
- `server/live_detector.py` — optional live YOLO26n: the dashboard sends JPEG frames (~2.5/s) and the backend detects with the same weights and threshold (0.25) as the WiMANS features, so models B and C get training-like inputs. Choose it under **Detectorul care alimentează fuziunea**. With COCO-SSD (default), B and C face a detector domain shift; the dashboard states which detector is active.
- Ultralytics YOLO is AGPL-3.0: check the licence before any commercial or closed-source use.

Endpoints: `GET /api/status`, `POST /api/csi`, `GET /api/benchmarks`, `POST /api/explain`, `WS /ws/live`.

## What this repository actually measures

**Experiment W (WiMANS)**: binary *human present vs no human present* on aligned CSI + video examples. This is intentionally simpler than WiMANS's original multi-user, multi-label activity recognition task. Three competing models use the **same training, validation and held-out test IDs**: CSI-only 1D-CNN; camera-only frozen YOLO-person detection features + MLP; late-fusion CNN+MLP. Do not report these binary results as activity recognition results.

**Experiment M (MM-Fi)**: CSI-only *sequence action recognition* with the officially released MM-Fi toolbox and annotation action codes. MM-Fi is NOT a binary presence dataset (people are present in action sequences), and RGB images are subject to additional access/availability conditions. We do not mix MM-Fi with WiMANS labels or claim an apples-to-apples cross-dataset F1 comparison. MM-Fi export is exploratory and action classes must be verified after prep.

**Qwen3-VL-2B-Instruct**: optional slow-loop description of a video frame and explicitly given real sensor evidence, **not** an automatically calibrated sensor-fusion network. Its output is unverified text and MUST NOT be used as a navigation control or safety alert.

## Installation (Python 3.11+ recommended)

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
pip install -r requirements.txt
# only for YOLO video extraction or Qwen:
pip install -r requirements-optional.txt
python -m pytest -q
```

## 0. Verify complete software pipeline without any real data

**No real-dataset F1/accuracy has been measured in this environment: you must obtain WiMANS/MM-Fi and run the relevant commands to generate it.**

This produces a **synthetic** signal fixture solely for software tests; it is NOT benchmark evidence for actual Wi-Fi sensing:

```bash
python make_synthetic_fixture.py
python train.py --data data/processed/synthetic_fixture.npz --mode all --epochs 3 --run-dir runs/synthetic_smoke
```

Report: `runs/synthetic_smoke/metrics.json`, with `"synthetic_fixture": true` prominently marked. It is prohibited to describe synthetic-test scores as published or real-dataset performance.

## 1. Download WiMANS yourself

**Kaggle update:** See [KAGGLE_SOURCES.md](KAGGLE_SOURCES.md) for three verified dataset links, download commands, and usage limitations. WiMANS is available via its author-provided Kaggle page; MM-Fi remains a separate official dataset download.

Official: https://github.com/huangshk/WiMANS

Dataset distribution is linked in its README, via Kaggle. Download and extract into `data/raw/WiMANS/`:

```text
data/raw/WiMANS/
    annotation.csv
    wifi_csi/amp/act_1_1.npy
    wifi_csi/mat/act_1_1.mat
    video/act_1_1.mp4
```

WiMANS documents **11,286** recordings, **3 seconds** each, **0–5 people**, and synchronized videos. The provided amplitude arrays (`*.npy`) are simplest for the initial model; this scaffold treats the first axis as time, matching the official benchmark's padding code. The CSI conversion conservatively summarizes antenna/subcarrier dimensions into 12 time-series channels to make varied radio formats accessible. This sacrifices potentially useful spatial signal structure; it is a **baseline**, not a published CSI architecture.

### Prepare (Wi-Fi only)

```bash
python prepare_wimans.py --root data/raw/WiMANS --output data/processed/wimans_presence.npz --limit 500
python train.py --data data/processed/wimans_presence.npz --mode csi --epochs 10 --run-dir runs/wimans_csi
```

The `--limit 500` command is for **quick iteration**. For a publishable benchmark, prepare the whole dataset by omitting `--limit` and ensure there are enough recording groups and both classes in each split.

### Prepare paired video + CSI and compare all three models

This requires `ultralytics` and compatible YOLO26n weights (downloaded the first time with internet access):

```bash
python prepare_wimans.py --root data/raw/WiMANS --with-yolo --weights yolo26n.pt \
  --output data/processed/wimans_paired.npz --limit 500
python train.py --data data/processed/wimans_paired.npz --mode all --epochs 10 \
  --run-dir runs/wimans_fusion
```

Camera features are deterministic summaries of YOLO's **person detections from sampled video frames** (mean/max count, confidence, left/center/right and bbox area). This is not an RGB action detector: a stationary frame cannot tell whether a person is walking. The encoder in `models.py` is deliberately minimal late-fusion.

## 2. MM-Fi separately (action recognition)

Official: https://github.com/ybhbingo/MMFi_dataset ; project page: https://ntu-aiot-lab.github.io/mm-fi

1. Obtain dataset under its access conditions. Do not redistribute photos or unlicensed material.
2. Clone the **official** code from the repository into `external/MMFi_dataset` yourself.
3. Copy `external/MMFi_dataset/config.yaml` to `configs/mmfi.yaml` and edit: `modality: wifi-csi`, `data_unit: sequence`. Select a suitable split; confirm subject disjointness.
4. Run adapter, then CSI-only multi-class training.

```bash
python prepare_mmfi.py --toolbox external/MMFi_dataset --root data/raw/MMFi_Dataset \
  --config configs/mmfi.yaml --output data/processed/mmfi_actions.npz --limit 200
python train.py --data data/processed/mmfi_actions.npz --mode csi --epochs 10 \
  --group-by groups --run-dir runs/mmfi_actions
```

The official toolbox returns `input_wifi-csi`, `action`, `scene`, and `subject`. A whole sequence is one sample; grouping on `subject` prevents sharing a subject between train/val/test. In the official loader the per-frame CSI array uses `CSIamp` from MAT files. Dataset access and `config.yaml` details must be verified locally. We do **not** use COCO labels for MM-Fi human action labels.

The MM-Fi action example is valid only if the resulting `class_names` and split distributions are sensible; class coverage may fail with `--limit 200`. For careful benchmark use full sequences and an explicit preregistered held-out subject split.

## 3. Qwen (optional, not real-time hazard logic)

```bash
python qwen_explain.py --image /path/to/real_frame.jpg \
  --evidence '{"camera_person_count":1,"radio_motion_detected":true,"radio_association":"unknown"}'
```

This downloads a large model; requires enough memory. For a 12h hackathon, use rule-based sensor explanations first. **Never** provide a claim of an invisible person from CSI alone. Quantitative hallucination benchmarking is a distinct project requiring audited ground truth.

## 4. Evaluation protocol (no leakage)

- Define the task **before** training. WiMANS first task: *is at least one person present?* Binary ground truth = `number_of_users > 0` from original annotation.
- Keep modality alignment by the common WiMANS sample `label`, e.g. `act_1_1` for `CSI .npy` and `video .mp4`.
- Group split **70/15/15 approximately** with `GroupShuffleSplit`, train vs validation vs unseen test. By default `groups` is derived from the sample label prefix (remove final underscore suffix); **verify experimentally whether that prefix truly corresponds to a recording/session**. For a stronger out-of-domain experiment use `--group-by environment`, but check that every split contains the relevant classes; three environments may be too few for stable conclusions.
- Never randomly split adjacent frames from one recording across splits.
- CSI normalization and visual standardization are fitted ONLY on training samples. A single model-specific decision threshold is chosen on validation, never test.
- Train all three models on **identical** paired samples and splits for a fair comparison.
- Report accuracy, macro-F1, positive-class precision and recall, balanced accuracy, confusion matrix, Brier (binary), test sample count, measured model forward p50/p95 latency, seed, split groups. For action recognition use macro-F1 and class confusion matrix. Benchmark real-world runtime separately, including YOLO and CSI capture.
- Report mean and standard deviation across at least **3 different predeclared seeds**, ideally additionally held-out environments and independent hardware settings. Do not cherry-pick the best seed.
- For a 95% confidence interval, cluster-bootstrap by independent recording/session or subject, not frame. Do not claim statistical significance just from one F1 score.
- Record CSI radio chipset/band/antennas. **WiMANS and MM-Fi radio data do not prove performance on Wi-Fi 6 802.11ax hardware**. ESP32-C6 / AX200 local data requires its own real-world validation and domain-shift tests.
- YOLO detections do not give true metric range or precise 3D space. This code does not validate mobility safety, blind navigation, brain stimulation or restored vision.

## 5. Training hyperparameters / practical schedule

Prototype: 10 epochs, AdamW, lr=1e-3, batch 32, 128 temporal steps, 12 derived CSI feature channels, BCEWithLogitsLoss for WiMANS presence, CrossEntropyLoss for MM-Fi action classification. These are **starting hyperparameters**, not published optimal settings.

1. Hour 1: datasets/labels + schema + small smoke sample.
2. Hours 2–3: CSI baseline, grouped validation, report.
3. Hours 4–5: frozen YOLO person features on **matching** videos.
4. Hours 6–7: fusion and 3-way comparison.
5. Hour 8: failures, ablation, missing modality behavior.
6. Hour 9: feed fused world-state into NeuroVision encoder (abstract digital events only).
7. Hour 10: optional Qwen grounded explanation (not real-time decisions).
8. Hours 11–12: data provenance, test outputs, demo/pitch.

## 6. Multi-seed reports

```bash
python run_seeds.py --data data/processed/wimans_paired.npz --mode all --epochs 10 --run-dir runs/wimans_three_seed --seeds 42,43,44
```

The wrapper runs the baseline separately for each seed and writes `runs/wimans_three_seed/aggregate.json`. It fails if an individual training run fails. Summary values should be reported as `mean ± sample standard deviation`, not as new measurements until these commands have actually run on the real downloaded dataset.

## 7. Provenance and reporting

`metrics.json` includes dataset name, whether synthetic, SHA-256 of prepared input, seed, groups, count per split, comparison scores and inference latency. Source weights and datasets are **not** bundled. Do not extrapolate performance to the blind, navigation or cortical visual implants.

### Trusted resources

- MM-Fi official: https://github.com/ybhbingo/MMFi_dataset
- MM-Fi site/license: https://ntu-aiot-lab.github.io/mm-fi
- WiMANS official: https://github.com/huangshk/WiMANS
- Ultralytics YOLO26: https://docs.ultralytics.com/models/yolo26
- Qwen3-VL-2B-Instruct: https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct

Check each individual source/license before commercial reuse. YOLO Ultralytics has AGPL/enterprise licensing considerations. Respect dataset privacy and recording consent.

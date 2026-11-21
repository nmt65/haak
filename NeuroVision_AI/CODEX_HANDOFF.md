# NeuroVision AI — predare către Codex

## Cum folosești acest fișier (în română)

1. Dezarhivează `NeuroVision_AI_complet.zip` și urcă folderul într-un repository GitHub (sau deschide-l local în Codex).
2. **Nu pune tokenul Kaggle în repository.** În Codex, adaugă-l ca *secret* / variabilă de mediu: `KAGGLE_USERNAME` și `KAGGLE_KEY` (din kaggle.json). Acceptă înainte termenii setului WiMANS pe kaggle.com.
3. Copiază în Codex promptul din secțiunea **„Prompt pentru Codex”** de mai jos, ca prim mesaj.
4. Cere-i lui Codex să lucreze pe rând, pe pașii din prompt, și să-ți arate rezultatele reale (fișierele `metrics.json` / `aggregate.csv`) înainte să treacă la următorul pas.

Ce funcționează deja și a fost verificat: un singur ecran live cu camera și YOLOE-26n-seg în browser (contur real din mască pentru orice obiect sau persoană, fără cutii sau contururi prestabilite), fosfenele pe același contur, harta direcțiilor, sunetul, adâncimea relativă (Depth Anything V2) și panoul Wi-Fi CSI. Panoul folosește backend-ul FastAPI: CSI simulat sau înregistrat, YOLO26n live și fuziunea. Modul demo, paginile de explicații, testul de ascultare, prezentarea și laboratoarele Codex au fost scoase intenționat. Ce **nu** s-a făcut încă: antrenarea pe datele reale WiMANS (lipsea tokenul Kaggle), MM-Fi și testarea cu hardware CSI real.

---

## Prompt pentru Codex

Copy everything below this line into Codex.

---

You are continuing work on **NeuroVision AI**, an assistive-technology research prototype (Life Science hackathon). The repository already contains a working React + TypeScript dashboard and a Python FastAPI backend with a PyTorch benchmark. **Do not start over and do not remove working functionality.** Work in small, verified steps and report real outputs.

### Hard rules

- Never invent metrics, dataset fields, labels or results. Every number you report must come from a file you produced by running code in this session.
- Results on synthetic data (`dataset: "SYNTHETIC"`, `synthetic_fixture: true`) are software checks only. Never present them as real performance.
- Keep the scientific honesty constraints already in the code and UI: the system does not stimulate the brain, does not restore vision, gives no metric distances without depth, never assigns a position to Wi-Fi CSI presence, and labels simulated data everywhere.
- Never commit secrets (`kaggle.json`, tokens, `.env`). Read Kaggle credentials from the `KAGGLE_USERNAME` / `KAGGLE_KEY` environment variables.
- The user interface language is Romanian, with correct diacritics (ă, â, î, ș, ț).
- The product is ONE live screen (camera → contour phosphenes, directions, sound, depth, Wi-Fi CSI panel). Do not re-add a demo mode, explanation pages, presentation mode or a settings panel.
- Outlines come ONLY from each object's segmentation mask (YOLOE). Never draw boxes, never fall back to a box border or any predefined/fitted shape, and never fill the inside of an outline in the phosphenes.
- Keep `src/services/yoloeClasses.json` and the ONNX model in sync: regenerate both with `neurovision_benchmark/export_browser_model.py`.
- Keep all tests passing: `pytest` in `neurovision_benchmark/` and `npm test` + `npm run build` at the root. Add tests for new code.
- Code style: descriptive names, short single-purpose functions, named constants, comments that explain why. Readability over cleverness.

### Repository layout

```
/                         React + Vite + TypeScript + Tailwind dashboard
  src/app/App.tsx         the single live screen
  src/context/            PipelineContext (camera pipeline), BackendContext (WebSocket to backend)
  src/services/           objectDetectionService (YOLOE, ONNX Runtime), yoloSegmentation (decoding), objectOutline (mask → outline), contourEncoder (outline → phosphenes), depth, sectors, audio
  public/models/          yoloe-26n-seg-pf-384.onnx (regenerate with neurovision_benchmark/export_browser_model.py)
  src/components/         header, camera, phosphenes, directions, CsiPanel (Wi-Fi CSI)
  tests/                  Vitest
neurovision_benchmark/    Python benchmark + live backend
  nv_benchmark/           CSI features, models (CSI 1D-CNN, camera MLP, late fusion), YOLO video features
  prepare_wimans.py       WiMANS -> .npz (binary presence: number_of_users > 0)
  prepare_mmfi.py         MM-Fi -> .npz (CSI action recognition, separate task)
  train.py, run_seeds.py  grouped splits, train-only normalisation, early stopping, metrics JSON + CSV
  train_wimans.ps1        Windows end-to-end script (download, prepare, train 3 seeds)
  server/                 FastAPI: CSI sources, model runtime, fusion world state, YOLO26n live (Qwen endpoint kept, not used by the UI)
  tests/                  pytest (15 tests)
  runs/                   git-ignored; on first start the backend trains the labelled synthetic smoke run itself
```

Read `README.md`, `neurovision_benchmark/README.md` and `neurovision_benchmark/KAGGLE_SOURCES.md` before changing anything.

### Environment setup (Linux)

```bash
# frontend
npm install
npm test && npm run build

# backend
cd neurovision_benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install -r requirements.txt -r requirements-server.txt "ultralytics>=8.4" kaggle
python -c "from ultralytics import YOLO; YOLO('yolo26n.pt')"   # downloads official YOLO26n weights (5.5 MB)
python -m pytest -q
```

(On Windows the project used a venv at `%LOCALAPPDATA%\neurovision\venv` because of the 260-character path limit.)

### Step 1 — Verify the existing state

Run all tests and the build. `runs/` and `data/` are git-ignored, so if they are missing the backend trains the labelled synthetic smoke run itself on first start; to do it by hand:

```bash
python make_synthetic_fixture.py
python train.py --data data/processed/synthetic_fixture.npz --mode all --epochs 3 --run-dir runs/synthetic_smoke
```

Start the backend (`python -m uvicorn server.app:app --port 8000` from `neurovision_benchmark/`) and confirm `GET /api/status` answers. Report any failure before continuing.

### Step 2 — WiMANS: download and verify the data

1. Download: `kaggle datasets download -d shuokanghuang/wimans -p data/downloads`.
2. **List the archive entries before extracting.** Verify the real structure: `annotation.csv`, `wifi_csi/amp/*.npy`, `video/*.mp4`. Print the columns of `annotation.csv` and the shape/dtype of a few `.npy` files. Check the dataset licence.
3. Extract under `data/raw/WiMANS/`. If the structure differs from what `prepare_wimans.py` expects, adapt the script to the verified structure — do not guess.
4. Verify that the label prefix used for grouping (`act_30_25` -> `act_30`) really corresponds to a recording/session. If it does not, use a verified grouping (e.g. environment or session column) to keep splits leakage-free.

### Step 3 — Train and benchmark (binary presence task)

1. CSI-only baseline first, quick pass: `python prepare_wimans.py --root <root> --output data/processed/wimans_presence.npz --limit 500`, then `python train.py --data data/processed/wimans_presence.npz --mode csi --epochs 10 --run-dir runs/wimans_csi`.
2. Paired camera + CSI with YOLO26n video features: `python prepare_wimans.py --root <root> --with-yolo --weights yolo26n.pt --output data/processed/wimans_paired.npz` (start with `--limit 500`, then the full dataset if time and compute allow).
3. Compare all three models on identical splits, three seeds: `python run_seeds.py --data data/processed/wimans_paired.npz --mode all --epochs 10 --seeds 42,43,44 --run-dir runs/wimans_three_seed`, plus one run in `runs/wimans_fusion` (the backend loads this folder automatically).
4. Report from the produced files: accuracy, precision, recall, macro-F1, balanced accuracy, false positive rate, confusion matrices, p50/p95 latency — mean ± sample standard deviation over the 3 seeds, test sample counts, class balance per split. State clearly if `--limit` was used.
5. If a split misses a class, do not report the run as a benchmark; fix the grouping or sample size.

### Step 4 — Multi-label activity recognition on WiMANS (if time permits)

Only with activity columns verified in `annotation.csv`. Add a multi-label task (BCE per activity, per-activity F1, macro-F1), keep the same grouped splits, and do not mix WiMANS activity labels with MM-Fi labels.

### Step 5 — MM-Fi (separate task, optional)

Follow section 2 of `neurovision_benchmark/README.md`: official toolbox, `modality: wifi-csi`, `data_unit: sequence`, subject-disjoint splits. Train a separate CSI action model and report it separately. Skip if access to MM-Fi is not available, and say so.

### Step 6 — Connect real results to the dashboard

1. Restart the backend; it prefers `runs/wimans_fusion`, then `runs/wimans_csi`, before the synthetic run. Confirm `/api/status` reports `trained_on_synthetic_data: false`.
2. Confirm the **Wi-Fi CSI** panel shows "Antrenate pe: WIMANS" and the synthetic-data warning disappears.
3. In **Înregistrat (set de date)** CSI mode the backend replays held-out WiMANS test samples with their true label; check that predictions and labels are displayed and plausible.

### Step 7 — Optional improvements (only after steps 1–6 work)

- CSI-Bench adapter (`guozhenjennzhu/csi-bench`) as an additional, separately reported CSI task.
- Validate the ESP32-CSI-Tool hardware mode with a real board, if available; record chipset, band and antennas.
- HTTPS dev server so the camera works on a phone (rear camera).
- If real depth estimates are active, optionally modulate the audio repetition rate by relative depth, with the assumption stated in the UI.

### Final report

At the end list: what you implemented, which models were trained and on which data, the actual evaluation results (with file paths), what still requires hardware or dataset access, and the exact commands to start the app.

---

## Note tehnice pentru tine

- **Pornire locală (Windows):** din folderul proiectului `npm install` apoi `npm run dev` (http://localhost:5173). Backend-ul, din `neurovision_benchmark`: creează venv-ul ca în README, apoi `python -m uvicorn server.app:app --port 8000`, și în aplicație **Wi-Fi CSI → Conectează**.
- **Modele:** YOLOE-26n-seg este inclus în `public/models` (15,6 MB, merge offline). Se descarcă la prima utilizare: Depth Anything V2 (în browser) și YOLO26n (5,5 MB, backend).
- **Licențe:** Ultralytics YOLO este AGPL-3.0; seturile de date au licențele lor (verifică înainte de folosire comercială).
- **Probleme întâlnite și rezolvate:** calea foarte lungă a folderului pe Windows strică instalarea pachetelor Python (de aceea venv-ul e într-o cale scurtă); Qwen are nevoie de `torchvision`; după modificarea `tailwind.config.js` serverul de dezvoltare trebuie repornit; în fundal browserul încetinește animațiile, deci FPS-ul măsurat scade când fereastra nu e vizibilă.

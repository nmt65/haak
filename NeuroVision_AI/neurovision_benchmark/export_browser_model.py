"""Exports the browser segmentation model used by the live screen.

YOLOE-26n-seg "prompt-free" (Ultralytics) finds and segments 4585 kinds of things.
It is exported to ONNX at 384×384 with NMS inside the model, so the browser only
has to read up to 300 finished rows. Writes:

  ../public/models/yoloe-26n-seg-pf-384.onnx   the model (~15 MB)
  ../src/services/yoloeClasses.json            its class names, in index order

Usage (from neurovision_benchmark/, with the venv active):
  pip install "ultralytics>=8.4" onnx onnxslim
  python export_browser_model.py
"""

import ast
import json
import shutil
from pathlib import Path

import onnx
from ultralytics import YOLO

WEIGHTS = "yoloe-26n-seg-pf.pt"  # downloaded by Ultralytics on first use
INPUT_SIZE = 384  # ~180 ms per frame on WebGPU in our tests; 480 was ~315 ms
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_TARGET = PROJECT_ROOT / "public" / "models" / f"yoloe-26n-seg-pf-{INPUT_SIZE}.onnx"
CLASSES_TARGET = PROJECT_ROOT / "src" / "services" / "yoloeClasses.json"


def read_class_names(onnx_path: Path) -> list[str]:
    """The class names are stored in the ONNX metadata as a Python dict literal {index: name}."""
    metadata = {prop.key: prop.value for prop in onnx.load(str(onnx_path)).metadata_props}
    names = ast.literal_eval(metadata["names"])
    return [names[index] for index in range(len(names))]


def main() -> None:
    exported = Path(YOLO(WEIGHTS).export(format="onnx", imgsz=INPUT_SIZE, simplify=True, opset=17, nms=True))
    MODEL_TARGET.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, MODEL_TARGET)

    class_names = read_class_names(exported)
    CLASSES_TARGET.write_text(json.dumps(class_names, ensure_ascii=False), encoding="utf-8")
    print(f"Model: {MODEL_TARGET} ({MODEL_TARGET.stat().st_size / 1e6:.1f} MB), {len(class_names)} classes")


if __name__ == "__main__":
    main()

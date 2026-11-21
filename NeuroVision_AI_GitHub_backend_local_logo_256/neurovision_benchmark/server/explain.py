"""Romanian scene explanations from the world state.

Two levels:
  * rule-based (always available, instant): built only from measured evidence;
  * Qwen3-VL-2B-Instruct (optional, slow on CPU): sees the frame + the same evidence.
    Its output is unverified generated text and must never drive navigation or alerts.
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time

LABELS_RO = {
    "person": "persoană", "chair": "scaun", "couch": "canapea", "dining table": "masă", "bottle": "sticlă",
    "cup": "cană", "laptop": "laptop", "tv": "televizor", "backpack": "rucsac", "bicycle": "bicicletă",
    "potted plant": "plantă", "bed": "pat", "cell phone": "telefon", "book": "carte", "dog": "câine",
}
SECTORS_RO = {"left": "în stânga", "center": "în centru", "right": "în dreapta"}
QWEN_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
QWEN_IMAGE_WIDTH = 448  # smaller image = fewer vision tokens = faster on CPU
QWEN_MAX_NEW_TOKENS = 120


def _describe_objects(objects: list[dict]) -> str:
    phrases = []
    for sector in ("left", "center", "right"):
        labels = [LABELS_RO.get(o["label"], o["label"]) for o in objects if o.get("sector") == sector]
        if labels:
            phrases.append(f"{', '.join(sorted(set(labels)))} {SECTORS_RO[sector]}")
    return "; ".join(phrases)


def rule_based_explanation(world: dict) -> str:
    """Deterministic description: states only what the sensors reported, plus their limits."""
    sentences = []
    camera = world.get("camera", {})
    if camera.get("is_fresh"):
        objects = _describe_objects(world.get("objects", []))
        sentences.append(f"Camera detectează: {objects}." if objects else "Camera nu detectează niciun obiect cunoscut în acest moment.")
    else:
        sentences.append("Nu există un cadru recent de la cameră.")

    radio = world.get("radio")
    csi = world.get("models", {}).get("csi")
    if radio is None:
        sentences.append("Nu există date radio Wi-Fi.")
    else:
        source = "simulate" if radio["is_simulated"] else ("înregistrate" if radio["mode"] == "recorded" else "în timp real")
        if csi is None:
            sentences.append(f"Datele radio ({source}) sunt disponibile, dar nu este încărcat un model CSI.")
        else:
            verdict = "indică prezență umană" if csi["is_present"] else "nu indică prezență umană"
            sentences.append(f"Datele radio ({source}) {verdict} (probabilitate {csi['probability']:.0%}), dar nu permit identificarea sursei sau a poziției.")

    agreement = world.get("fusion", {}).get("agreement")
    if agreement == "camera_person_radio_absent":
        sentences.append("Camera și radioul nu sunt de acord: rezultatul este incert.")
    elif agreement == "radio_present_camera_none":
        sentences.append("Radioul sugerează prezență fără confirmare vizuală; persoana poate fi în afara câmpului camerei.")

    unknown = world.get("unknown_sectors", [])
    if unknown:
        sentences.append(f"Zone fără detecții ({', '.join(SECTORS_RO[s] for s in unknown)}) sunt necunoscute, nu libere.")
    return " ".join(sentences)


def grounded_evidence(world: dict) -> dict:
    """The only facts Qwen is allowed to rely on besides the image."""
    csi = world.get("models", {}).get("csi")
    radio = world.get("radio") or {}
    return {
        "camera_objects": [{"obiect": LABELS_RO.get(o["label"], o["label"]), "zona": SECTORS_RO.get(o["sector"], "?"),
                            "incredere": round(float(o["confidence"]), 2)} for o in world.get("objects", [])],
        "radio_presence_probability": None if csi is None else csi["probability"],
        "radio_data_kind": radio.get("mode"),
        "radio_is_simulated": radio.get("is_simulated"),
        "radio_motion_detected": radio.get("motion_detected"),
        "radio_position_information": "niciuna",
        "camera_radio_agreement": world.get("fusion", {}).get("agreement"),
    }


class QwenExplainer:
    """Loads Qwen3-VL-2B lazily (first request) and serialises generation on one thread."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.status = "neîncărcat"
        self.error: str | None = None
        self.lock = threading.Lock()

    def _load(self):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self.status = "se încarcă"
        # Load both parts first and only keep them if BOTH succeed, so a failed load can be retried cleanly.
        processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
        # float32 on CPU: slower than bf16 on some CPUs but numerically safe everywhere.
        model = Qwen3VLForConditionalGeneration.from_pretrained(QWEN_MODEL_ID, dtype=torch.float32, device_map="cpu")
        self.processor, self.model = processor, model
        self.status = "gata"

    def explain(self, frame_data_url: str, world: dict) -> dict:
        from PIL import Image

        with self.lock:
            try:
                if self.model is None:
                    self._load()
                image = decode_data_url(frame_data_url)
                image.thumbnail((QWEN_IMAGE_WIDTH, QWEN_IMAGE_WIDTH))
                prompt = (
                    "Descrie concis, în cel mult două propoziții în limba română, doar obiectele vizibile și dovezile "
                    "senzorilor de mai jos. Nu afirma că Wi-Fi a identificat persoana din imagine. Nu inventa obiecte, "
                    "distanțe sau persoane invizibile. Menționează incertitudinea. Dovezi structurate:\n"
                    + json.dumps(grounded_evidence(world), ensure_ascii=False)
                )
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}]
                inputs = self.processor.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
                )
                start = time.perf_counter()
                output = self.model.generate(**inputs, max_new_tokens=QWEN_MAX_NEW_TOKENS, do_sample=False)
                latency_s = time.perf_counter() - start
                new_tokens = [o[len(i):] for i, o in zip(inputs.input_ids, output)]
                text = self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
                self.error = None
                return {"text": text, "latency_s": round(latency_s, 1)}
            except Exception as exc:  # report the real failure to the dashboard
                self.status = "eroare"
                self.error = f"{type(exc).__name__}: {exc}"
                raise


def decode_data_url(data_url: str):
    from PIL import Image

    encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")

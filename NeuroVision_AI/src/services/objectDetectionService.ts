import type { DetectedObject } from "../types/vision";
import { assignHorizontalSector, normalizeBoundingBox } from "../utils/coordinateTransforms";
import { buildObjectMask, computeLetterbox, decodeCandidates, removeDuplicateObjects, ROW_LENGTH, type ObjectMask } from "./yoloSegmentation";
import YOLOE_CLASSES from "./yoloeClasses.json";
// The WebAssembly runtime is served from this app (not a CDN), so detection also works offline.
import ortWasmUrl from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";

/*
 * Object detection AND segmentation in one model: YOLOE-26n-seg "prompt-free",
 * run locally in the browser with ONNX Runtime (WebGPU when available, otherwise
 * WebAssembly). It knows 4585 kinds of things and returns, for each one, a box and
 * a pixel mask, so every object gets its real outline. Camera frames never leave the browser.
 */

/** One object found in a frame: box [x, y, width, height] in frame pixels, its name, score and pixel mask. */
export interface RawDetection {
  bbox: [number, number, number, number];
  class: string;
  score: number;
  mask: ObjectMask;
}

export interface ObjectDetector {
  device: "webgpu" | "wasm";
  /** Finds objects in the current image of a video or canvas of the given size. */
  detect(source: CanvasImageSource, frameWidth: number, frameHeight: number): Promise<RawDetection[]>;
}

/** Exported from Ultralytics by neurovision_benchmark/export_browser_model.py. */
const MODEL_URL = "/models/yoloe-26n-seg-pf-384.onnx";
const MODEL_INPUT_SIZE = 384;
/** The grey the model was trained with for padding (114 / 255). */
const LETTERBOX_GREY = "rgb(114, 114, 114)";
/** Objects below this score are not even decoded; the pipeline applies its own, higher threshold. */
const MIN_DECODE_SCORE = 0.25;
/** Two boxes overlapping more than this are the same object with two names. */
const DUPLICATE_IOU = 0.6;
const MAX_DETECTIONS = 20;
/** Masks are built at this width across the frame: twice the model's own mask resolution, for smoother outlines. */
const MASK_WIDTH = 192;

/** Romanian display names for common labels (anything else keeps the model's English name). */
export const LABEL_TRANSLATIONS: Record<string, string> = {
  person: "persoană",
  bicycle: "bicicletă",
  car: "mașină",
  motorcycle: "motocicletă",
  bus: "autobuz",
  truck: "camion",
  dog: "câine",
  cat: "pisică",
  chair: "scaun",
  couch: "canapea",
  bench: "bancă",
  bed: "pat",
  "dining table": "masă",
  "potted plant": "plantă",
  suitcase: "valiză",
  backpack: "rucsac",
  bottle: "sticlă",
  cup: "cană",
  tv: "televizor",
  laptop: "laptop",
  "cell phone": "telefon",
  book: "carte",
  keyboard: "tastatură",
  mouse: "mouse",
  refrigerator: "frigider",
  toilet: "toaletă",
  umbrella: "umbrelă",
  handbag: "geantă",
};

export function translateLabel(label: string): string {
  return LABEL_TRANSLATIONS[label] ?? label;
}

let detectorPromise: Promise<ObjectDetector> | null = null;

/** Draws the frame scaled into the square model input, padded with grey, as RGB floats 0–1 (channel by channel). */
function prepareInput(canvas: HTMLCanvasElement, source: CanvasImageSource, frameWidth: number, frameHeight: number) {
  const letterbox = computeLetterbox(frameWidth, frameHeight, MODEL_INPUT_SIZE);
  canvas.width = MODEL_INPUT_SIZE;
  canvas.height = MODEL_INPUT_SIZE;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("Canvas 2D indisponibil pentru detecție.");
  context.fillStyle = LETTERBOX_GREY;
  context.fillRect(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE);
  context.drawImage(source, letterbox.padX, letterbox.padY, frameWidth * letterbox.scale, frameHeight * letterbox.scale);

  const rgba = context.getImageData(0, 0, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE).data;
  const planeSize = MODEL_INPUT_SIZE * MODEL_INPUT_SIZE;
  const pixels = new Float32Array(3 * planeSize);
  for (let pixel = 0; pixel < planeSize; pixel++) {
    pixels[pixel] = rgba[pixel * 4] / 255;
    pixels[planeSize + pixel] = rgba[pixel * 4 + 1] / 255;
    pixels[2 * planeSize + pixel] = rgba[pixel * 4 + 2] / 255;
  }
  return { letterbox, pixels };
}

async function createDetector(): Promise<ObjectDetector> {
  const ort = await import("onnxruntime-web/webgpu");
  ort.env.wasm.wasmPaths = { wasm: ortWasmUrl };
  ort.env.logLevel = "error"; // ONNX Runtime otherwise logs harmless placement warnings as console errors

  let device: "webgpu" | "wasm" = "gpu" in navigator ? "webgpu" : "wasm";
  let session;
  try {
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [device] });
  } catch (webGpuError) {
    if (device !== "webgpu") throw webGpuError;
    console.warn("WebGPU indisponibil pentru detecție, se folosește WebAssembly:", webGpuError);
    device = "wasm";
    session = await ort.InferenceSession.create(MODEL_URL, { executionProviders: [device] });
  }

  const inputCanvas = document.createElement("canvas");
  return {
    device,
    async detect(source, frameWidth, frameHeight) {
      const { letterbox, pixels } = prepareInput(inputCanvas, source, frameWidth, frameHeight);
      const outputs = await session.run({ images: new ort.Tensor("float32", pixels, [1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE]) });
      const objects = outputs.output0.data as Float32Array;
      const prototypes = outputs.output1.data as Float32Array;
      const protoSize = outputs.output1.dims[3];

      const candidates = decodeCandidates(objects, objects.length / ROW_LENGTH, MIN_DECODE_SCORE);
      return removeDuplicateObjects(candidates, DUPLICATE_IOU, MAX_DETECTIONS).map((candidate) => {
        const mask = buildObjectMask(candidate, prototypes, protoSize, letterbox, MASK_WIDTH);
        const left = (candidate.box.left - letterbox.padX) / letterbox.scale;
        const top = (candidate.box.top - letterbox.padY) / letterbox.scale;
        const width = (candidate.box.right - candidate.box.left) / letterbox.scale;
        const height = (candidate.box.bottom - candidate.box.top) / letterbox.scale;
        return { bbox: [left, top, width, height], class: YOLOE_CLASSES[candidate.classIndex] ?? "obiect", score: candidate.score, mask };
      });
    },
  };
}

/** Loads the model once (~15 MB, then cached by the browser). A failed load can be retried. */
export function loadObjectDetector(): Promise<ObjectDetector> {
  if (!detectorPromise) {
    detectorPromise = createDetector().catch((error: unknown) => {
      detectorPromise = null; // allow a retry
      throw error;
    });
  }
  return detectorPromise;
}

/**
 * Names the model uses for one person. They all become "person", so people are counted
 * the same way everywhere (sound, Wi-Fi CSI fusion). "crowd" is a group, so it stays.
 */
const PERSON_LABELS = new Set(["man", "woman", "boy", "girl", "child", "baby", "toddler", "teenager", "adult", "player", "businessman", "pedestrian"]);

export function normalizeLabel(label: string): string {
  return PERSON_LABELS.has(label) ? "person" : label;
}

/** Turns raw model output into normalized DetectedObjects, in the same order. */
export function toDetectedObjects(rawDetections: RawDetection[], imageWidth: number, imageHeight: number, isMirrored: boolean): DetectedObject[] {
  return rawDetections.map((raw) => {
    const bbox = normalizeBoundingBox(raw.bbox, imageWidth, imageHeight, isMirrored);
    return {
      id: "",
      label: normalizeLabel(raw.class),
      confidence: raw.score,
      bbox,
      horizontalSector: assignHorizontalSector(bbox),
      relativeDepth: null,
      source: "camera",
    };
  });
}

/** Max distance (normalized) between box centers to treat two detections as the same object. */
const TRACKING_MAX_CENTER_DISTANCE = 0.15;

function centerDistance(a: DetectedObject, b: DetectedObject): number {
  const deltaX = a.bbox.x + a.bbox.width / 2 - (b.bbox.x + b.bbox.width / 2);
  const deltaY = a.bbox.y + a.bbox.height / 2 - (b.bbox.y + b.bbox.height / 2);
  return Math.hypot(deltaX, deltaY);
}

/**
 * Very simple tracking: a new detection keeps the id of the closest previous
 * detection with the same label. Otherwise it gets a fresh id.
 */
export function assignStableIds(
  previous: DetectedObject[],
  current: DetectedObject[],
  createId: () => string,
): DetectedObject[] {
  const usedIds = new Set<string>();
  return current.map((detection) => {
    let bestMatch: DetectedObject | null = null;
    let bestDistance = TRACKING_MAX_CENTER_DISTANCE;
    for (const candidate of previous) {
      if (candidate.label !== detection.label || usedIds.has(candidate.id)) continue;
      const distance = centerDistance(candidate, detection);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestMatch = candidate;
      }
    }
    const id = bestMatch ? bestMatch.id : createId();
    usedIds.add(id);
    return { ...detection, id };
  });
}

/** A box edge closer than this (fraction of the frame) to the frame edge counts as touching it. */
const FRAME_EDGE_MARGIN = 0.03;
/** Surfaces are flat bands; anything taller than this fraction of the frame is kept. */
const SURFACE_MAX_HEIGHT = 0.7;
/** A region at least this wide and this tall (fractions of the frame) is the whole scene ("rock concert", "room"). */
const WHOLE_SCENE_FRACTION = 0.85;

/**
 * The model also names surfaces ("city square", "floor", "sky") and whole scenes ("rock concert").
 * A surface stretches from the left edge of the image to the right edge and continues past the
 * top or bottom edge; a scene covers almost the whole frame. Their outline is a horizon or the
 * frame itself and would hide the objects in the phosphenes, so they are skipped.
 * A wide object (a bus) does not reach the top or bottom; a person close to the camera is tall.
 */
export function isBackgroundSurface(raw: RawDetection, frameWidth: number, frameHeight: number): boolean {
  const [x, y, width, height] = raw.bbox;
  const marginX = frameWidth * FRAME_EDGE_MARGIN;
  const marginY = frameHeight * FRAME_EDGE_MARGIN;
  const touchesLeftAndRight = x <= marginX && x + width >= frameWidth - marginX;
  const touchesTopOrBottom = y <= marginY || y + height >= frameHeight - marginY;
  const isSurface = touchesLeftAndRight && touchesTopOrBottom && height < frameHeight * SURFACE_MAX_HEIGHT;
  const isWholeScene = width >= frameWidth * WHOLE_SCENE_FRACTION && height >= frameHeight * WHOLE_SCENE_FRACTION;
  return isSurface || isWholeScene;
}

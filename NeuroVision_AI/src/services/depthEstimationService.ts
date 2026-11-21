import type { DetectedObject } from "../types/vision";
import { clamp } from "../utils/coordinateTransforms";

/*
 * Optional RELATIVE monocular depth estimation with Depth Anything V2 (small),
 * running locally in the browser through Transformers.js (WebGPU when available,
 * otherwise WebAssembly).
 *
 * The model output is relative: it only says which parts of THIS image are
 * nearer or farther than others. It is never converted into meters.
 */

const DEPTH_MODEL_ID = "onnx-community/depth-anything-v2-small";

/** A grayscale depth map: one byte per pixel, 255 = nearest, 0 = farthest in this frame. */
export interface DepthMap {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

export interface DepthEstimator {
  device: "webgpu" | "wasm";
  estimate(input: HTMLCanvasElement): Promise<DepthMap>;
}

export type DepthLoadProgress = (percent: number) => void;

let estimatorPromise: Promise<DepthEstimator> | null = null;

function isWebGpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

async function createEstimator(onProgress?: DepthLoadProgress): Promise<DepthEstimator> {
  const transformers = await import("@huggingface/transformers");

  // Transformers.js reports download progress per file; we forward the overall percentage.
  const progressCallback = (event: { status: string; progress?: number }) => {
    if (event.status === "progress" && typeof event.progress === "number") onProgress?.(event.progress);
  };

  const loadOn = (device: "webgpu" | "wasm") =>
    transformers.pipeline("depth-estimation", DEPTH_MODEL_ID, {
      device,
      dtype: device === "webgpu" ? "fp16" : "q8", // smaller quantized weights on CPU
      progress_callback: progressCallback,
    });

  let device: "webgpu" | "wasm" = isWebGpuAvailable() ? "webgpu" : "wasm";
  let estimator;
  try {
    estimator = await loadOn(device);
  } catch (webGpuError) {
    if (device !== "webgpu") throw webGpuError;
    console.warn("WebGPU indisponibil pentru adâncime, se folosește WebAssembly:", webGpuError);
    device = "wasm";
    estimator = await loadOn(device);
  }

  return {
    device,
    estimate: async (input) => {
      const image = transformers.RawImage.fromCanvas(input);
      const output = (await estimator(image)) as unknown as { depth: DepthMap };
      return { width: output.depth.width, height: output.depth.height, data: output.depth.data };
    },
  };
}

/** Loads the depth model once (≈25–50 MB on first use, then cached by the browser). */
export function loadDepthEstimator(onProgress?: DepthLoadProgress): Promise<DepthEstimator> {
  if (!estimatorPromise) {
    estimatorPromise = createEstimator(onProgress).catch((error: unknown) => {
      estimatorPromise = null; // allow a retry
      throw error;
    });
  }
  return estimatorPromise;
}

/** Pixels sampled per box side when reading the depth map (keeps it fast). */
const SAMPLES_PER_SIDE = 12;
/** Use the nearer part of the object (75th percentile of "nearness"), not its background edges. */
const NEARNESS_PERCENTILE = 0.75;

/**
 * Reads the depth map inside one normalized box and returns relative depth:
 * 0 = among the nearest things in this frame, 1 = among the farthest.
 */
export function measureRelativeDepth(depthMap: DepthMap, box: DetectedObject["bbox"]): number | null {
  const nearnessValues: number[] = [];
  for (let stepY = 0; stepY < SAMPLES_PER_SIDE; stepY++) {
    for (let stepX = 0; stepX < SAMPLES_PER_SIDE; stepX++) {
      const normalizedX = box.x + ((stepX + 0.5) / SAMPLES_PER_SIDE) * box.width;
      const normalizedY = box.y + ((stepY + 0.5) / SAMPLES_PER_SIDE) * box.height;
      const pixelX = clamp(Math.floor(normalizedX * depthMap.width), 0, depthMap.width - 1);
      const pixelY = clamp(Math.floor(normalizedY * depthMap.height), 0, depthMap.height - 1);
      nearnessValues.push(depthMap.data[pixelY * depthMap.width + pixelX]);
    }
  }
  if (nearnessValues.length === 0) return null;

  nearnessValues.sort((a, b) => a - b);
  const index = Math.min(nearnessValues.length - 1, Math.floor(nearnessValues.length * NEARNESS_PERCENTILE));
  const nearness = nearnessValues[index] / 255;
  return Math.round((1 - nearness) * 1000) / 1000;
}

/**
 * Attaches relative depth to each detection. The depth map must come from the
 * same (possibly mirrored) view as the boxes, so pass `isMirrored` to flip it.
 */
export function attachRelativeDepth(detections: DetectedObject[], depthMap: DepthMap | null, isMirrored: boolean): DetectedObject[] {
  if (!depthMap) return detections;
  return detections.map((detection) => {
    const box = isMirrored ? { ...detection.bbox, x: 1 - detection.bbox.x - detection.bbox.width } : detection.bbox;
    return { ...detection, relativeDepth: measureRelativeDepth(depthMap, box) };
  });
}

/** Converts relative depth into a short, honest word. Never meters. */
export function describeRelativeDepth(relativeDepth: number | null): string {
  if (relativeDepth === null) return "adâncime necunoscută";
  if (relativeDepth < 0.34) return "mai aproape";
  if (relativeDepth < 0.67) return "intermediar";
  return "mai departe";
}

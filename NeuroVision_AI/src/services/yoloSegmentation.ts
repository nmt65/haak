import { clamp } from "../utils/coordinateTransforms";

/*
 * Decoding the output of YOLOE-26n-seg "prompt-free": instance segmentation for
 * 4585 kinds of things (people, furniture, cups, shoes, plants, signs …), so almost
 * anything put in front of the camera gets its own mask.
 * Pure functions, no browser APIs, so every step can be unit-tested.
 *
 * The model sees a square image: the camera frame scaled down and padded with grey
 * ("letterbox"). It returns two tensors:
 *   output0 [1, 300, 38]: up to 300 objects, each = box (left, top, right, bottom in input
 *            pixels), score, class index and 32 mask coefficients; unused rows have score 0
 *   output1 [1, 32, P, P]: 32 "prototype" masks shared by all objects
 * An object's mask = its 32 coefficients × the 32 prototypes, summed per pixel.
 */

/** Values per object row in output0: left, top, right, bottom, score, class, then the mask coefficients. */
const ROW_SCORE = 4;
const ROW_CLASS = 5;
const ROW_COEFFICIENTS_START = 6;
export const MASK_COEFFICIENTS = 32;
export const ROW_LENGTH = ROW_COEFFICIENTS_START + MASK_COEFFICIENTS;

/** How the camera frame was placed inside the square model input. */
export interface Letterbox {
  frameWidth: number;
  frameHeight: number;
  inputSize: number;
  scale: number; // input pixels per frame pixel
  padX: number; // grey border on the left (input pixels)
  padY: number; // grey border on top (input pixels)
}

/** Box in pixels: [left, right) × [top, bottom). */
export interface PixelBox {
  left: number;
  top: number;
  right: number;
  bottom: number;
}

export interface Candidate {
  classIndex: number;
  score: number;
  box: PixelBox; // in model input pixels
  coefficients: Float32Array;
}

/** Pixels of one object (1) or not (0), covering the whole camera frame. Frame orientation, never mirrored. */
export interface ObjectMask {
  width: number;
  height: number;
  data: Uint8Array;
}

/** Scales the frame to fit the square input without distortion and centres it. */
export function computeLetterbox(frameWidth: number, frameHeight: number, inputSize: number): Letterbox {
  const scale = inputSize / Math.max(frameWidth, frameHeight);
  const padX = Math.floor((inputSize - Math.round(frameWidth * scale)) / 2);
  const padY = Math.floor((inputSize - Math.round(frameHeight * scale)) / 2);
  return { frameWidth, frameHeight, inputSize, scale, padX, padY };
}

/** Reads every object row whose score reaches `minScore`. */
export function decodeCandidates(output: Float32Array, rowCount: number, minScore: number): Candidate[] {
  const candidates: Candidate[] = [];
  for (let row = 0; row < rowCount; row++) {
    const offset = row * ROW_LENGTH;
    const score = output[offset + ROW_SCORE];
    if (score < minScore) continue;
    candidates.push({
      classIndex: Math.round(output[offset + ROW_CLASS]),
      score,
      box: { left: output[offset], top: output[offset + 1], right: output[offset + 2], bottom: output[offset + 3] },
      coefficients: output.slice(offset + ROW_COEFFICIENTS_START, offset + ROW_LENGTH),
    });
  }
  return candidates;
}

function boxArea(box: PixelBox): number {
  return Math.max(0, box.right - box.left) * Math.max(0, box.bottom - box.top);
}

/** Intersection over union: 0 = no overlap, 1 = identical boxes. */
export function intersectionOverUnion(a: PixelBox, b: PixelBox): number {
  const overlapWidth = Math.min(a.right, b.right) - Math.max(a.left, b.left);
  const overlapHeight = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
  if (overlapWidth <= 0 || overlapHeight <= 0) return 0;
  const overlap = overlapWidth * overlapHeight;
  return overlap / (boxArea(a) + boxArea(b) - overlap);
}

/**
 * With thousands of classes the model often names one object several ways
 * (a bus is also "recreational vehicle"). Keeps the highest-scoring box and drops
 * any box, whatever its class, that overlaps it too much.
 */
export function removeDuplicateObjects(candidates: Candidate[], iouThreshold: number, maxCount: number): Candidate[] {
  const sorted = [...candidates].sort((a, b) => b.score - a.score);
  const kept: Candidate[] = [];
  for (const candidate of sorted) {
    if (kept.length >= maxCount) break;
    const isDuplicate = kept.some((keptCandidate) => intersectionOverUnion(keptCandidate.box, candidate.box) > iouThreshold);
    if (!isDuplicate) kept.push(candidate);
  }
  return kept;
}

/** Converts a box from model input pixels to camera frame pixels, clipped to the frame. */
export function inputBoxToFrame(box: PixelBox, letterbox: Letterbox): PixelBox {
  const toFrameX = (x: number) => clamp((x - letterbox.padX) / letterbox.scale, 0, letterbox.frameWidth);
  const toFrameY = (y: number) => clamp((y - letterbox.padY) / letterbox.scale, 0, letterbox.frameHeight);
  return { left: toFrameX(box.left), top: toFrameY(box.top), right: toFrameX(box.right), bottom: toFrameY(box.bottom) };
}

/**
 * Builds one object's mask at `maskWidth` pixels across the frame.
 * The prototype grid (a quarter of the input resolution) is coarse, so the per-pixel
 * mask value is interpolated between prototype pixels before deciding in or out:
 * that gives a smooth outline instead of a staircase. Only pixels inside the
 * object's box can belong to it (as in the original YOLO post-processing).
 */
export function buildObjectMask(candidate: Candidate, prototypes: Float32Array, protoSize: number, letterbox: Letterbox, maskWidth: number): ObjectMask {
  const maskHeight = Math.max(1, Math.round((maskWidth * letterbox.frameHeight) / letterbox.frameWidth));
  const data = new Uint8Array(maskWidth * maskHeight);
  const frameBox = inputBoxToFrame(candidate.box, letterbox);
  const maskPerFrame = maskWidth / letterbox.frameWidth;
  const protoPerInput = protoSize / letterbox.inputSize;

  // Mask value (before the sigmoid) of one prototype pixel; > 0 means "object".
  const logitCache = new Map<number, number>();
  const logitAt = (protoX: number, protoY: number) => {
    const x = clamp(protoX, 0, protoSize - 1);
    const y = clamp(protoY, 0, protoSize - 1);
    const pixel = y * protoSize + x;
    let logit = logitCache.get(pixel);
    if (logit === undefined) {
      logit = 0;
      for (let k = 0; k < MASK_COEFFICIENTS; k++) logit += candidate.coefficients[k] * prototypes[k * protoSize * protoSize + pixel];
      logitCache.set(pixel, logit);
    }
    return logit;
  };

  const firstX = Math.floor(frameBox.left * maskPerFrame);
  const lastX = Math.min(maskWidth, Math.ceil(frameBox.right * maskPerFrame));
  const firstY = Math.floor(frameBox.top * maskPerFrame);
  const lastY = Math.min(maskHeight, Math.ceil(frameBox.bottom * maskPerFrame));

  for (let maskY = firstY; maskY < lastY; maskY++) {
    for (let maskX = firstX; maskX < lastX; maskX++) {
      // Centre of this mask pixel → frame → model input → prototype grid (continuous coordinates).
      const inputX = ((maskX + 0.5) / maskPerFrame) * letterbox.scale + letterbox.padX;
      const inputY = ((maskY + 0.5) / maskPerFrame) * letterbox.scale + letterbox.padY;
      const protoX = inputX * protoPerInput - 0.5;
      const protoY = inputY * protoPerInput - 0.5;

      const x0 = Math.floor(protoX);
      const y0 = Math.floor(protoY);
      const fractionX = protoX - x0;
      const fractionY = protoY - y0;
      const top = logitAt(x0, y0) * (1 - fractionX) + logitAt(x0 + 1, y0) * fractionX;
      const bottom = logitAt(x0, y0 + 1) * (1 - fractionX) + logitAt(x0 + 1, y0 + 1) * fractionX;
      if (top * (1 - fractionY) + bottom * fractionY > 0) data[maskY * maskWidth + maskX] = 1;
    }
  }
  return { width: maskWidth, height: maskHeight, data };
}

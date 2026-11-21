import { clamp } from "../utils/coordinateTransforms";
import type { ObjectMask } from "./yoloSegmentation";

/*
 * ObjectOutline — turns each object's pixel mask into its outline.
 *
 *   mask (1 = object pixel) → light clean-up → boundary pixels
 *
 * The same outline is drawn on the camera image and turned into phosphenes,
 * so both windows show exactly the same shape. Nothing here is a predefined
 * shape: an object without mask pixels simply has no outline.
 */

/** Pixel indices (y × width + x) of an outline inside a width×height frame. Frame orientation, never mirrored. */
export interface PixelOutline {
  width: number;
  height: number;
  pixels: number[];
}

/** A pixel stays part of the object after clean-up when at least this many of its 3×3 neighbourhood (9 pixels) are. */
const MAJORITY_COUNT = 5;

/**
 * Majority filter: removes isolated specks and fills pinholes, so the outline
 * follows the object instead of the model's noise.
 */
export function cleanMask(mask: ObjectMask): ObjectMask {
  const { width, height, data } = mask;
  const cleaned = new Uint8Array(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let objectCount = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          // Outside the frame repeats the nearest edge pixel, so the frame edge does not erode the object.
          const sampleX = clamp(x + dx, 0, width - 1);
          const sampleY = clamp(y + dy, 0, height - 1);
          objectCount += data[sampleY * width + sampleX];
        }
      }
      cleaned[y * width + x] = objectCount >= MAJORITY_COUNT ? 1 : 0;
    }
  }
  return { width, height, data: cleaned };
}

/**
 * Boundary pixels: object pixels with at least one background neighbour (left, right, up, down).
 * An object cut off by the frame edge gets no line along that edge (outside counts as object).
 */
export function findMaskBoundary(mask: ObjectMask): Uint8Array {
  const { width, height, data } = mask;
  const boundary = new Uint8Array(data.length);
  const isObject = (x: number, y: number) => x < 0 || y < 0 || x >= width || y >= height || data[y * width + x] === 1;

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[y * width + x] !== 1) continue;
      const touchesBackground = !isObject(x - 1, y) || !isObject(x + 1, y) || !isObject(x, y - 1) || !isObject(x, y + 1);
      if (touchesBackground) boundary[y * width + x] = 1;
    }
  }
  return boundary;
}

/** The outline of one object, or null when its mask is empty after clean-up. */
export function outlineFromMask(mask: ObjectMask): PixelOutline | null {
  const boundary = findMaskBoundary(cleanMask(mask));
  const pixels: number[] = [];
  for (let index = 0; index < boundary.length; index++) {
    if (boundary[index] === 1) pixels.push(index);
  }
  return pixels.length > 0 ? { width: mask.width, height: mask.height, pixels } : null;
}

/** Outline of every object, keyed by detection id. Objects whose mask is empty are left out. */
export function buildObjectOutlines(masks: ReadonlyMap<string, ObjectMask>): Map<string, PixelOutline> {
  const outlines = new Map<string, PixelOutline>();
  for (const [id, mask] of masks) {
    const outline = outlineFromMask(mask);
    if (outline) outlines.set(id, outline);
  }
  return outlines;
}

import type { HorizontalSector, NormalizedBox } from "../types/vision";

/** The image is split into three equal vertical bands: left, center, right. */
const LEFT_SECTOR_END = 1 / 3;
const RIGHT_SECTOR_START = 2 / 3;

/** Keeps a value inside the range [min, max]. */
export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Converts a pixel bounding box [x, y, width, height] into 0–1 coordinates.
 * The box is clipped so it never leaves the image.
 * If `isMirrored` is true the box is flipped horizontally, so it matches a
 * mirrored (selfie-style) preview.
 */
export function normalizeBoundingBox(
  pixelBox: [number, number, number, number],
  imageWidth: number,
  imageHeight: number,
  isMirrored = false,
): NormalizedBox {
  if (imageWidth <= 0 || imageHeight <= 0) {
    return { x: 0, y: 0, width: 0, height: 0 };
  }

  const [pixelX, pixelY, pixelWidth, pixelHeight] = pixelBox;
  const left = clamp(pixelX / imageWidth, 0, 1);
  const top = clamp(pixelY / imageHeight, 0, 1);
  const right = clamp((pixelX + pixelWidth) / imageWidth, 0, 1);
  const bottom = clamp((pixelY + pixelHeight) / imageHeight, 0, 1);

  const width = right - left;
  const height = bottom - top;
  const x = isMirrored ? 1 - right : left;

  return { x, y: top, width, height };
}

/** Assigns a box to left / center / right using the horizontal position of its center. */
export function assignHorizontalSector(box: NormalizedBox): HorizontalSector {
  const centerX = box.x + box.width / 2;
  if (centerX < LEFT_SECTOR_END) return "left";
  if (centerX >= RIGHT_SECTOR_START) return "right";
  return "center";
}

/**
 * Maps a horizontal image position (0–1) to a viewing angle in degrees,
 * assuming a pinhole camera with the given horizontal field of view.
 * Negative = left of straight ahead, positive = right.
 */
export function imageXToAngle(normalizedX: number, fieldOfViewDegrees: number): number {
  const halfFovRadians = ((fieldOfViewDegrees / 2) * Math.PI) / 180;
  const offsetFromCenter = (normalizedX - 0.5) * 2; // -1 … 1
  const angleRadians = Math.atan(offsetFromCenter * Math.tan(halfFovRadians));
  return (angleRadians * 180) / Math.PI;
}

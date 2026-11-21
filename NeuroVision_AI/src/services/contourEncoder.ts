import type { DetectedObject } from "../types/vision";
import { clamp } from "../utils/coordinateTransforms";
import type { PixelOutline } from "./objectOutline";

/*
 * ContourEncoder — the phosphene encoder of the app. Phosphenes light up along the
 * OUTLINE of each detected object (from its segmentation mask), never filling it.
 *
 * Idea (used in simulated prosthetic vision studies): with very few phosphenes,
 * an object's silhouette is easier to recognise than a filled blob.
 *
 *   object outline pixels → count per phosphene cell → activation 0–1
 */

/** A cell lights only when the outline crosses at least this fraction of its side (removes specks). */
const MIN_CELL_COVERAGE = 0.25;
/** Outline covering this fraction of a cell's side already gives full brightness. */
const FULL_BRIGHTNESS_COVERAGE = 0.5;
/** Brightness of an object at 0 % and 100 % detector confidence. */
const MIN_OBJECT_WEIGHT = 0.55;
const MAX_OBJECT_WEIGHT = 1;
/** With relative depth, the farthest objects are dimmed by this much (nearer = brighter). */
const FAR_OBJECT_DIMMING = 0.4;

/** Brighter for confident and (when depth is known) nearer objects. */
function objectWeight(detection: DetectedObject): number {
  let weight = MIN_OBJECT_WEIGHT + (MAX_OBJECT_WEIGHT - MIN_OBJECT_WEIGHT) * clamp(detection.confidence, 0, 1);
  if (detection.relativeDepth !== null) weight *= 1 - FAR_OBJECT_DIMMING * clamp(detection.relativeDepth, 0, 1);
  return weight;
}

/** Size of the phosphene matrix: columns across the image, rows down it. */
export interface MatrixSize {
  columns: number;
  rows: number;
}

/** Counts outline pixels per phosphene cell, in on-screen orientation. Returns rows × columns. */
function countOutlinePixelsPerCell(outline: PixelOutline, size: MatrixSize, isMirrored: boolean): number[][] {
  const counts = Array.from({ length: size.rows }, () => new Array<number>(size.columns).fill(0));
  for (const pixelIndex of outline.pixels) {
    const frameX = pixelIndex % outline.width;
    const y = Math.floor(pixelIndex / outline.width);
    const screenX = isMirrored ? outline.width - 1 - frameX : frameX;
    const col = Math.min(size.columns - 1, Math.floor((screenX / outline.width) * size.columns));
    const row = Math.min(size.rows - 1, Math.floor((y / outline.height) * size.rows));
    counts[row][col] += 1;
  }
  return counts;
}

/** Adds one outline to the matrix. Overlapping objects keep the brighter value per cell instead of adding up. */
function addOutlineToMatrix(matrix: number[][], outline: PixelOutline, weight: number, isMirrored: boolean): void {
  const size = { rows: matrix.length, columns: matrix[0]?.length ?? 0 };
  // A straight outline crossing a cell covers about one cell side of pixels.
  const cellSidePx = (outline.width / size.columns + outline.height / size.rows) / 2;
  const counts = countOutlinePixelsPerCell(outline, size, isMirrored);
  for (let row = 0; row < size.rows; row++) {
    for (let col = 0; col < size.columns; col++) {
      const coverage = counts[row][col] / cellSidePx;
      if (coverage < MIN_CELL_COVERAGE) continue;
      const activation = clamp(coverage / FULL_BRIGHTNESS_COVERAGE, 0, 1) * weight;
      matrix[row][col] = Math.max(matrix[row][col], activation);
    }
  }
}

/**
 * Builds the phosphene activation matrix (rows × columns, values 0–1) from the
 * outline of each detection (keyed by detection id). A detection without an
 * outline lights nothing: no box or placeholder shape is ever drawn instead.
 */
export function buildContourMatrix(
  detections: DetectedObject[],
  outlines: ReadonlyMap<string, PixelOutline>,
  size: MatrixSize,
  isMirrored: boolean,
): number[][] {
  const matrix = Array.from({ length: size.rows }, () => new Array<number>(size.columns).fill(0));
  for (const detection of detections) {
    const outline = outlines.get(detection.id);
    if (outline) addOutlineToMatrix(matrix, outline, objectWeight(detection), isMirrored);
  }
  return matrix;
}

/** One frame of simulated phosphene activations (0–1). Software values only, not a stimulation protocol. */
export interface PhospheneFrame {
  columns: number;
  rows: number;
  activations: number[][];
}

/** Activations below this are dropped, so faint specks do not flicker on and off. */
const ACTIVATION_THRESHOLD = 0.12;

/** Turns a contour matrix into a phosphene frame: weak cells off, values rounded to 3 decimals. */
export function toPhospheneFrame(matrix: number[][]): PhospheneFrame {
  const activations = matrix.map((matrixRow) =>
    matrixRow.map((value) => {
      const activation = clamp(value, 0, 1);
      return activation < ACTIVATION_THRESHOLD ? 0 : Math.round(activation * 1000) / 1000;
    }),
  );
  return { columns: matrix[0]?.length ?? 0, rows: matrix.length, activations };
}

/** A frame with every phosphene off. */
export function createEmptyPhospheneFrame(size: MatrixSize): PhospheneFrame {
  return toPhospheneFrame(Array.from({ length: size.rows }, () => new Array<number>(size.columns).fill(0)));
}

/**
 * Columns are fixed; rows follow the camera's aspect ratio so every phosphene cell is square
 * in the image and a silhouette keeps its proportions (16:9 → 24 × 14, 4:3 → 24 × 18).
 */
export function matrixSizeForFrame(columns: number, frameWidth: number, frameHeight: number): MatrixSize {
  if (frameWidth <= 0 || frameHeight <= 0) return { columns, rows: columns };
  return { columns, rows: Math.max(1, Math.round((columns * frameHeight) / frameWidth)) };
}

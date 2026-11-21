import { describe, expect, it } from "vitest";
import { buildObjectOutlines, cleanMask, findMaskBoundary, outlineFromMask } from "../src/services/objectOutline";
import { buildContourMatrix, toPhospheneFrame } from "../src/services/contourEncoder";
import { assignHorizontalSector } from "../src/utils/coordinateTransforms";
import type { ObjectMask } from "../src/services/yoloSegmentation";
import type { DetectedObject } from "../src/types/vision";

const MASK_SIZE = 160;
const GRID_SIZE = 16; // 10 px per cell
const MATRIX = { columns: GRID_SIZE, rows: GRID_SIZE };

/** Mask with object pixels in the given rectangles [left, top, right, bottom). */
function makeMask(rectangles: [number, number, number, number][]): ObjectMask {
  const data = new Uint8Array(MASK_SIZE * MASK_SIZE);
  for (const [left, top, right, bottom] of rectangles) {
    for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) data[y * MASK_SIZE + x] = 1;
  }
  return { width: MASK_SIZE, height: MASK_SIZE, data };
}

function makeDetection(id: string, x: number, y: number, width: number, height: number): DetectedObject {
  const bbox = { x, y, width, height };
  return { id, label: "chair", confidence: 1, bbox, horizontalSector: assignHorizontalSector(bbox), relativeDepth: null, source: "camera" };
}

/** A thin upright object (pixels 70–90 wide, 20–150 tall) inside a much wider detector box. */
const THIN_OBJECT: [number, number, number, number] = [70, 20, 90, 150];
const WIDE_BOX = makeDetection("a", 0.25, 0.1, 0.5, 0.85);

/** An "L" shape: a box around it would be a rectangle, the outline must not be. */
const L_SHAPE: [number, number, number, number][] = [
  [40, 40, 60, 120],
  [40, 100, 120, 120],
];

function litCells(matrix: number[][]): Set<string> {
  const cells = new Set<string>();
  matrix.forEach((matrixRow, row) => matrixRow.forEach((value, col) => value > 0 && cells.add(`${row},${col}`)));
  return cells;
}

describe("object outline from a mask", () => {
  it("marks only the border of the object", () => {
    const boundary = findMaskBoundary(makeMask([THIN_OBJECT]));
    expect(boundary[50 * MASK_SIZE + 70]).toBe(1); // left side
    expect(boundary[50 * MASK_SIZE + 89]).toBe(1); // right side
    expect(boundary[50 * MASK_SIZE + 80]).toBe(0); // inside
    expect(boundary[50 * MASK_SIZE + 40]).toBe(0); // background
  });

  it("removes isolated specks before tracing", () => {
    const cleaned = cleanMask(makeMask([THIN_OBJECT, [10, 10, 11, 11]]));
    expect(cleaned.data[10 * MASK_SIZE + 10]).toBe(0);
    expect(cleaned.data[80 * MASK_SIZE + 80]).toBe(1);
  });

  it("gives no outline at all for an empty mask (never a placeholder shape)", () => {
    expect(outlineFromMask(makeMask([]))).toBeNull();
    expect(buildObjectOutlines(new Map([["a", makeMask([])]])).size).toBe(0);
  });
});

describe("contour phosphenes", () => {
  it("follow the object, not its detector box", () => {
    const outlines = buildObjectOutlines(new Map([["a", makeMask([THIN_OBJECT])]]));
    const matrix = buildContourMatrix([WIDE_BOX], outlines, MATRIX, false);

    expect(matrix[8][7]).toBeGreaterThan(0.5); // left side of the object (pixel 70)
    expect(matrix[8][8]).toBeGreaterThan(0.5); // right side of the object (pixel 89)
    expect(matrix[8][4]).toBe(0); // where the box border would be (pixel 40)
    expect(matrix[8][11]).toBe(0);
  });

  it("keep the inside dark and trace any shape", () => {
    const outlines = buildObjectOutlines(new Map([["a", makeMask(L_SHAPE)]]));
    const matrix = buildContourMatrix([makeDetection("a", 0.25, 0.25, 0.5, 0.5)], outlines, MATRIX, false);
    const cells = litCells(matrix);

    expect(cells.has("6,4")).toBe(true); // vertical bar of the L
    expect(cells.has("11,8")).toBe(true); // bottom bar of the L
    expect(cells.has("6,9")).toBe(false); // empty corner a box would fill
  });

  it("light nothing for a detection without an outline", () => {
    const matrix = buildContourMatrix([WIDE_BOX], new Map(), MATRIX, false);
    expect(litCells(matrix).size).toBe(0);
  });

  it("follow the mirrored selfie view", () => {
    // Object on the LEFT of the camera frame appears on the RIGHT of a mirrored screen.
    const outlines = buildObjectOutlines(new Map([["a", makeMask([[10, 20, 40, 150]])]]));
    const matrix = buildContourMatrix([makeDetection("a", 0.7, 0.1, 0.3, 0.85)], outlines, MATRIX, true);
    const columns = [...litCells(matrix)].map((cell) => Number(cell.split(",")[1]));
    expect(columns.length).toBeGreaterThan(0);
    expect(Math.min(...columns)).toBeGreaterThanOrEqual(GRID_SIZE / 2);
  });

  it("become a phosphene frame without filling the inside again", () => {
    const outlines = buildObjectOutlines(new Map([["a", makeMask([[45, 45, 115, 115]])]]));
    const frame = toPhospheneFrame(buildContourMatrix([WIDE_BOX], outlines, MATRIX, false));
    expect(frame.activations[7][7]).toBe(0);
    expect(frame.activations[7][4]).toBeGreaterThan(0.5);
  });
});

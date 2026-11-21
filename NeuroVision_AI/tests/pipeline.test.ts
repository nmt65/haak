import { describe, expect, it, vi } from "vitest";
import { assignHorizontalSector, normalizeBoundingBox } from "../src/utils/coordinateTransforms";
import { computeSectorIntensities } from "../src/services/sectorService";
import { buildContourMatrix, matrixSizeForFrame, toPhospheneFrame } from "../src/services/contourEncoder";
import { buildObjectOutlines } from "../src/services/objectOutline";
import { stopCameraStream } from "../src/services/cameraService";
import { isBackgroundSurface, normalizeLabel, toDetectedObjects } from "../src/services/objectDetectionService";
import type { DetectedObject } from "../src/types/vision";

/** Helper: one detection with a given normalized box. */
function makeDetection(x: number, y: number, width: number, height: number, confidence = 1): DetectedObject {
  const bbox = { x, y, width, height };
  return { id: "test", label: "chair", confidence, bbox, horizontalSector: assignHorizontalSector(bbox), relativeDepth: null, source: "camera" };
}

describe("1. bounding box normalization", () => {
  it("converts pixels to 0–1 coordinates", () => {
    expect(normalizeBoundingBox([64, 48, 320, 240], 640, 480)).toEqual({ x: 0.1, y: 0.1, width: 0.5, height: 0.5 });
  });

  it("clips boxes that leave the image", () => {
    const box = normalizeBoundingBox([-100, 400, 300, 200], 640, 480);
    expect(box.x).toBe(0);
    expect(box.y + box.height).toBeCloseTo(1);
  });

  it("mirrors horizontally when requested", () => {
    const box = normalizeBoundingBox([0, 0, 160, 480], 640, 480, true);
    expect(box.x).toBeCloseTo(0.75);
    expect(box.width).toBeCloseTo(0.25);
  });

  it("returns an empty box for an invalid image size", () => {
    expect(normalizeBoundingBox([0, 0, 10, 10], 0, 0)).toEqual({ x: 0, y: 0, width: 0, height: 0 });
  });
});

describe("2. horizontal sector assignment", () => {
  it("uses the box center", () => {
    expect(assignHorizontalSector({ x: 0.05, y: 0, width: 0.2, height: 1 })).toBe("left");
    expect(assignHorizontalSector({ x: 0.4, y: 0, width: 0.2, height: 1 })).toBe("center");
    expect(assignHorizontalSector({ x: 0.75, y: 0, width: 0.2, height: 1 })).toBe("right");
  });

  it("counts every name for one person as a person", () => {
    expect(normalizeLabel("woman")).toBe("person");
    expect(normalizeLabel("child")).toBe("person");
    expect(normalizeLabel("crowd")).toBe("crowd");
    expect(normalizeLabel("chair")).toBe("chair");
  });

  it("skips surfaces and whole-scene regions but keeps wide or tall objects", () => {
    const mask = { width: 1, height: 1, data: new Uint8Array(1) };
    const make = (bbox: [number, number, number, number]) => ({ bbox, class: "x", score: 0.9, mask });
    expect(isBackgroundSurface(make([0, 260, 480, 220]), 480, 480)).toBe(true); // ground band at the bottom
    expect(isBackgroundSurface(make([5, 100, 470, 220]), 480, 480)).toBe(false); // bus: wide, floats in the middle
    expect(isBackgroundSurface(make([60, 40, 360, 440]), 480, 480)).toBe(false); // person close to the camera: tall
    expect(isBackgroundSurface(make([2, 3, 476, 470]), 480, 480)).toBe(true); // the whole scene
  });

  it("converts raw detections in order and assigns sectors", () => {
    const mask = { width: 1, height: 1, data: new Uint8Array(1) };
    const raw = [
      { bbox: [0, 0, 100, 100] as [number, number, number, number], class: "person", score: 0.9, mask },
      { bbox: [500, 0, 100, 100] as [number, number, number, number], class: "chair", score: 0.6, mask },
    ];
    const detections = toDetectedObjects(raw, 640, 480, false);
    expect(detections.map((detection) => detection.horizontalSector)).toEqual(["left", "right"]);
    expect(detections[0].source).toBe("camera");
  });
});

/** 120×90 mask (4:3) with one filled rectangle [left, top, right, bottom) in pixels. */
function makeOutlines(rectangle: [number, number, number, number]) {
  const width = 120;
  const height = 90;
  const data = new Uint8Array(width * height);
  const [left, top, right, bottom] = rectangle;
  for (let y = top; y < bottom; y++) for (let x = left; x < right; x++) data[y * width + x] = 1;
  return buildObjectOutlines(new Map([["test", { width, height, data }]]));
}

describe("3. phosphene matrix size", () => {
  it("keeps the camera's proportions with square cells", () => {
    expect(matrixSizeForFrame(24, 640, 480)).toEqual({ columns: 24, rows: 18 });
    expect(matrixSizeForFrame(24, 1280, 720)).toEqual({ columns: 24, rows: 14 });
    expect(matrixSizeForFrame(24, 0, 0)).toEqual({ columns: 24, rows: 24 });
  });
});

describe("4. activation thresholding", () => {
  it("drops faint cells and keeps every activation between 0 and 1", () => {
    const frame = toPhospheneFrame([
      [0.05, 0.5],
      [1.4, -0.2],
    ]);
    expect(frame.activations).toEqual([
      [0, 0.5],
      [1, 0],
    ]);
    expect(frame).toMatchObject({ columns: 2, rows: 2 });
  });
});

describe("5. phosphene encoding determinism", () => {
  it("returns identical output for identical input", () => {
    const outlines = makeOutlines([30, 20, 90, 70]);
    const detections = [makeDetection(0.2, 0.15, 0.6, 0.7, 0.8)];
    const size = matrixSizeForFrame(24, 120, 90);
    const first = toPhospheneFrame(buildContourMatrix(detections, outlines, size, false));
    const second = toPhospheneFrame(buildContourMatrix(detections, outlines, size, false));
    expect(second).toEqual(first);
    expect(first.activations.flat().some((value) => value > 0)).toBe(true);
  });
});

describe("7. empty detection handling", () => {
  it("produces no phosphenes and silent sectors", () => {
    const frame = toPhospheneFrame(buildContourMatrix([], makeOutlines([30, 20, 90, 70]), { columns: 24, rows: 18 }, false));
    expect(frame.activations.flat().every((value) => value === 0)).toBe(true);
    expect(computeSectorIntensities([])).toEqual({ left: 0, center: 0, right: 0 });
  });
});

describe("8. camera cleanup", () => {
  it("stops every track of the stream", () => {
    const tracks = [{ stop: vi.fn() }, { stop: vi.fn() }];
    const stream = { getTracks: () => tracks } as unknown as MediaStream;
    stopCameraStream(stream);
    expect(tracks.every((track) => track.stop.mock.calls.length === 1)).toBe(true);
  });

  it("does nothing when there is no stream", () => {
    expect(() => stopCameraStream(null)).not.toThrow();
  });
});

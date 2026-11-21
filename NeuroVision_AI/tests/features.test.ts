import { describe, expect, it } from "vitest";
import { attachRelativeDepth, describeRelativeDepth, measureRelativeDepth, type DepthMap } from "../src/services/depthEstimationService";
import { computeSectorIntensities } from "../src/services/sectorService";
import { assignHorizontalSector } from "../src/utils/coordinateTransforms";
import type { DetectedObject } from "../src/types/vision";

/** A 10×10 depth map whose left half is near (255) and right half is far (0). */
function makeSplitDepthMap(): DepthMap {
  const width = 10;
  const height = 10;
  const data = new Uint8Array(width * height);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) data[y * width + x] = x < 5 ? 255 : 0;
  return { width, height, data };
}

function makeDetection(label: string, x: number, width: number, confidence = 0.9): DetectedObject {
  const bbox = { x, y: 0.2, width, height: 0.6 };
  return { id: label, label, confidence, bbox, horizontalSector: assignHorizontalSector(bbox), relativeDepth: null, source: "camera" };
}

const CHAIR_LEFT = makeDetection("chair", 0.05, 0.2);
const PERSON_CENTER = makeDetection("person", 0.42, 0.16);

describe("relative depth", () => {
  it("reads near and far regions from the depth map", () => {
    const depthMap = makeSplitDepthMap();
    expect(measureRelativeDepth(depthMap, { x: 0, y: 0, width: 0.4, height: 1 })).toBe(0);
    expect(measureRelativeDepth(depthMap, { x: 0.6, y: 0, width: 0.4, height: 1 })).toBe(1);
  });

  it("flips the lookup when the preview is mirrored", () => {
    const depthMap = makeSplitDepthMap();
    const [mirrored] = attachRelativeDepth([CHAIR_LEFT], depthMap, true); // left of the mirrored view
    expect(mirrored.relativeDepth).toBe(1); // on the raw frame it sits on the far (right) half
  });

  it("leaves detections untouched without a depth map", () => {
    const detections = [CHAIR_LEFT, PERSON_CENTER];
    expect(attachRelativeDepth(detections, null, false)).toBe(detections);
  });

  it("describes depth in words, never meters", () => {
    expect(describeRelativeDepth(null)).toBe("adâncime necunoscută");
    expect(describeRelativeDepth(0.1)).toBe("mai aproape");
    expect(describeRelativeDepth(0.9)).toBe("mai departe");
  });
});

describe("sector intensities", () => {
  it("gives every occupied sector a value and leaves empty ones silent", () => {
    const intensities = computeSectorIntensities([CHAIR_LEFT, PERSON_CENTER]);
    expect(intensities.left).toBeGreaterThan(0);
    expect(intensities.center).toBeGreaterThan(0);
    expect(intensities.right).toBe(0);
  });
});

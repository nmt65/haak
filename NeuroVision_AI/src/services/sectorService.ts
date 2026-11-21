import type { DetectedObject, HorizontalSector } from "../types/vision";
import { clamp } from "../utils/coordinateTransforms";

/*
 * Sector summary: how strongly the left, center and right of the view are occupied.
 * Drives the spatial audio, the direction gauges and the screen-reader announcements.
 */

/** Strongest observation per horizontal sector (0 = nothing observed there). */
export type SectorIntensities = Record<HorizontalSector, number>;

/** Square root of the box area is multiplied by this, so a box covering ~40 % of the image counts as fully prominent. */
const SIZE_GAIN = 1.6;
const MIN_SIZE_FACTOR = 0.3;

/**
 * How prominent an object is, used for audio loudness.
 * Bigger in the image = more prominent. This is NOT a distance estimate.
 */
function computeProminence(detection: DetectedObject): number {
  const area = detection.bbox.width * detection.bbox.height;
  const sizeFactor = clamp(Math.sqrt(area) * SIZE_GAIN, MIN_SIZE_FACTOR, 1);
  return detection.confidence * sizeFactor;
}

/** Strongest prominence per horizontal sector (0 = nothing observed there). */
export function computeSectorIntensities(detections: DetectedObject[]): SectorIntensities {
  const intensities: SectorIntensities = { left: 0, center: 0, right: 0 };
  for (const detection of detections) {
    const sector = detection.horizontalSector;
    intensities[sector] = Math.max(intensities[sector], computeProminence(detection));
  }
  return intensities;
}

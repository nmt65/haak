/** Where a piece of data came from. The product only uses the live camera. */
export type InputSource = "camera";

export type HorizontalSector = "left" | "center" | "right";

/** Bounding box in normalized image coordinates (0–1, origin top-left). */
export interface NormalizedBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface DetectedObject {
  id: string;
  label: string;
  confidence: number;
  bbox: NormalizedBox;
  horizontalSector: HorizontalSector;
  /** Relative depth (0 = near, 1 = far). Null when no depth model is active. */
  relativeDepth: number | null;
  source: InputSource;
}

export type ModelStatus = "idle" | "loading" | "ready" | "error";

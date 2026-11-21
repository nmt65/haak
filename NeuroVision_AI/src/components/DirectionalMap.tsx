import { usePipeline } from "../context/PipelineContext";
import { useAnimatedCanvas } from "../hooks/useAnimatedCanvas";
import { imageXToAngle } from "../utils/coordinateTransforms";
import { translateLabel } from "../services/objectDetectionService";
import type { DetectedObject } from "../types/vision";

/*
 * DirectionalMap — a top-down view centred on the user, drawn like a
 * perimetry (visual-field) chart in ink on paper.
 *
 * Direction comes from the object's horizontal position in the image.
 * - Without depth: the distance is unknown, so the object is a hatched band
 *   across the whole range (never a fake point at a fake distance).
 * - With relative depth: the object sits on a nearer or farther ring.
 *   This is ordering only — no meters.
 */

/** Assumed horizontal field of view of a typical laptop webcam. An approximation. */
export const ASSUMED_FOV_DEGREES = 60;

const INK = "#F8FAFC";
const GRAPHITE = "#94A3B8";
const RULE = "#2A3A57";
const COBALT = "#8B9CF6";
const FONT = "'Atkinson Hyperlegible Next Variable', sans-serif";

const RANGE_RING_COUNT = 3;
const UNKNOWN_BAND_INNER = 0.22; // fraction of map radius
const UNKNOWN_BAND_OUTER = 0.97;
const DEPTH_BAND_HALF_THICKNESS = 0.07;
const ANGLE_TICKS_DEGREES = [-30, -20, -10, 0, 10, 20, 30];

interface MapGeometry {
  originX: number;
  originY: number;
  radius: number;
}

/** Converts a viewing angle (0 = straight ahead) into a canvas angle (radians). */
function toCanvasAngle(angleDegrees: number): number {
  return ((angleDegrees - 90) * Math.PI) / 180;
}

function pointAt(geometry: MapGeometry, angleDegrees: number, radiusFraction: number): [number, number] {
  const angle = toCanvasAngle(angleDegrees);
  return [geometry.originX + Math.cos(angle) * geometry.radius * radiusFraction, geometry.originY + Math.sin(angle) * geometry.radius * radiusFraction];
}

function traceWedge(context: CanvasRenderingContext2D, geometry: MapGeometry, fromDeg: number, toDeg: number, innerFraction: number, outerFraction: number): void {
  const { originX, originY, radius } = geometry;
  context.beginPath();
  context.arc(originX, originY, radius * outerFraction, toCanvasAngle(fromDeg), toCanvasAngle(toDeg));
  context.arc(originX, originY, radius * innerFraction, toCanvasAngle(toDeg), toCanvasAngle(fromDeg), true);
  context.closePath();
}

/** The chart frame: field edges, sector boundaries, range rings and angle ticks. */
function drawChart(context: CanvasRenderingContext2D, geometry: MapGeometry, hasDepth: boolean): void {
  const halfFov = ASSUMED_FOV_DEGREES / 2;

  context.lineWidth = 1;
  context.strokeStyle = RULE;
  for (let ring = 1; ring <= RANGE_RING_COUNT; ring++) {
    context.beginPath();
    context.arc(geometry.originX, geometry.originY, (geometry.radius * ring) / RANGE_RING_COUNT, toCanvasAngle(-halfFov), toCanvasAngle(halfFov));
    context.stroke();
  }

  // Sector boundaries (image thirds) as dashed rays.
  context.setLineDash([3, 4]);
  for (const boundary of [imageXToAngle(1 / 3, ASSUMED_FOV_DEGREES), imageXToAngle(2 / 3, ASSUMED_FOV_DEGREES)]) {
    const [endX, endY] = pointAt(geometry, boundary, 1);
    context.beginPath();
    context.moveTo(geometry.originX, geometry.originY);
    context.lineTo(endX, endY);
    context.stroke();
  }
  context.setLineDash([]);

  // Field-of-view edges in ink.
  context.strokeStyle = INK;
  for (const edge of [-halfFov, halfFov]) {
    const [endX, endY] = pointAt(geometry, edge, 1);
    context.beginPath();
    context.moveTo(geometry.originX, geometry.originY);
    context.lineTo(endX, endY);
    context.stroke();
  }

  // Angle ticks along the outer edge.
  context.font = `500 11px ${FONT}`;
  context.fillStyle = GRAPHITE;
  context.textAlign = "center";
  for (const tick of ANGLE_TICKS_DEGREES) {
    const [innerX, innerY] = pointAt(geometry, tick, 1);
    const [outerX, outerY] = pointAt(geometry, tick, 1.04);
    const [labelX, labelY] = pointAt(geometry, tick, 1.1);
    context.strokeStyle = INK;
    context.beginPath();
    context.moveTo(innerX, innerY);
    context.lineTo(outerX, outerY);
    context.stroke();
    context.fillText(`${tick}°`, labelX, labelY + 4);
  }

  if (hasDepth) {
    // Written along the straight-ahead line, next to the nearest and farthest rings.
    const [nearX, nearY] = pointAt(geometry, 0, 0.22);
    const [farX, farY] = pointAt(geometry, 0, 0.9);
    context.textAlign = "left";
    context.fillText("mai aproape", nearX + 6, nearY);
    context.fillText("mai departe", farX + 6, farY);
  }
}

/** Diagonal hatching inside the current path: marks "direction known, distance unknown". */
function fillHatched(context: CanvasRenderingContext2D, color: string): void {
  context.save();
  context.clip();
  context.strokeStyle = color;
  context.lineWidth = 1.2;
  const { width, height } = context.canvas;
  for (let offset = -height; offset < width; offset += 7) {
    context.beginPath();
    context.moveTo(offset, height);
    context.lineTo(offset + height, 0);
    context.stroke();
  }
  context.restore();
}

function drawObject(context: CanvasRenderingContext2D, geometry: MapGeometry, detection: DetectedObject): void {
  const fromDeg = imageXToAngle(detection.bbox.x, ASSUMED_FOV_DEGREES);
  const toDeg = imageXToAngle(detection.bbox.x + detection.bbox.width, ASSUMED_FOV_DEGREES);
  const middleDeg = (fromDeg + toDeg) / 2;

  let labelRadius: number;
  if (detection.relativeDepth === null) {
    traceWedge(context, geometry, fromDeg, toDeg, UNKNOWN_BAND_INNER, UNKNOWN_BAND_OUTER);
    fillHatched(context, "rgba(139, 156, 246, 0.45)");
    traceWedge(context, geometry, fromDeg, toDeg, UNKNOWN_BAND_INNER, UNKNOWN_BAND_OUTER);
    context.strokeStyle = COBALT;
    context.lineWidth = 1.5;
    context.stroke();
    labelRadius = 0.6;
  } else {
    const ringPosition = 0.2 + detection.relativeDepth * 0.72;
    traceWedge(context, geometry, fromDeg, toDeg, ringPosition - DEPTH_BAND_HALF_THICKNESS, ringPosition + DEPTH_BAND_HALF_THICKNESS);
    context.fillStyle = COBALT;
    context.shadowColor = COBALT;
    context.shadowBlur = 12;
    context.fill();
    context.shadowBlur = 0;
    labelRadius = ringPosition + DEPTH_BAND_HALF_THICKNESS + 0.08;
  }

  const [labelX, labelY] = pointAt(geometry, middleDeg, labelRadius);
  const text = translateLabel(detection.label);
  context.font = `700 12px ${FONT}`;
  const textWidth = context.measureText(text).width;
  context.fillStyle = "rgba(16, 24, 39, 0.92)";
  context.fillRect(labelX - textWidth / 2 - 4, labelY - 12, textWidth + 8, 17);
  context.fillStyle = INK;
  context.textAlign = "center";
  context.fillText(text, labelX, labelY + 1);
}

function drawUser(context: CanvasRenderingContext2D, geometry: MapGeometry): void {
  const { originX, originY } = geometry;
  context.fillStyle = INK;
  context.beginPath();
  context.arc(originX, originY, 7, 0, Math.PI * 2);
  context.fill();
  context.strokeStyle = INK;
  context.lineWidth = 1;
  context.beginPath();
  context.arc(originX, originY, 12, 0, Math.PI * 2);
  context.stroke();
  context.font = `600 12px ${FONT}`;
  context.fillStyle = INK;
  context.textAlign = "center";
  context.fillText("tu", originX, originY + 28);
}

/** Below this map radius the chart is skipped for the frame (arc() rejects negative radii). */
const MIN_DRAWABLE_RADIUS_PX = 40;

export function DirectionalMap() {
  const { snapshotRef } = usePipeline();

  const canvasRef = useAnimatedCanvas((context, width, height) => {
    const snapshot = snapshotRef.current;
    context.clearRect(0, 0, width, height);

    const bottomSpace = 40;
    const halfFovRadians = ((ASSUMED_FOV_DEGREES / 2) * Math.PI) / 180;
    const radiusByHeight = height - bottomSpace - 26;
    const radiusByWidth = width / (2 * Math.sin(halfFovRadians) * 1.15);
    const geometry: MapGeometry = { originX: width / 2, originY: height - bottomSpace, radius: Math.min(radiusByHeight, radiusByWidth) };
    // While the layout is still settling the canvas can be tiny; the rings' radii would go negative.
    if (geometry.radius < MIN_DRAWABLE_RADIUS_PX) return;

    const detections = snapshot.detections;
    const hasDepth = detections.some((detection) => detection.relativeDepth !== null);

    drawChart(context, geometry, hasDepth);
    for (const detection of detections) drawObject(context, geometry, detection);
    drawUser(context, geometry);
  });

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Hartă de sus, centrată pe utilizator: obiectele sunt plasate după direcție. Fără adâncime, distanța nu este estimată."
      className="block h-full w-full"
    />
  );
}

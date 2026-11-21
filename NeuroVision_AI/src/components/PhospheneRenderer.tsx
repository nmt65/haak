import { useRef } from "react";
import { usePipeline } from "../context/PipelineContext";
import { useAnimatedCanvas } from "../hooks/useAnimatedCanvas";

/*
 * PhospheneRenderer — draws the phosphene matrix as soft white-cyan points of light
 * on black, over faint visual-field axes (like a perimetry chart).
 * The matrix has the camera's proportions and is fitted like the camera image,
 * so a silhouette sits in the same place in both windows.
 * Purely illustrative: NOT a validated prediction of cortical prosthetic vision.
 */

/** Glow radius as a fraction of the cell size: neighbouring glows touch, so an outline reads as a line. */
const GLOW_RADIUS = 0.75;
/** Max random offset of a phosphene from its ideal position (fraction of a cell). */
const POSITION_JITTER = 0.18;
/** How fast a phosphene brightens toward a higher activation (ms for ~63% of the change). */
const RISE_TIME_MS = 70;
/** How long a phosphene keeps glowing after its activation disappears (ms for ~63% of the fade). */
const PERSISTENCE_MS = 450;
const FLICKER_DEPTH = 0.12;
const FLICKER_SPEED = 0.012;
const NOISE_SPECKS_PER_FRAME = 12;
const MIN_VISIBLE_INTENSITY = 0.01;
const AXIS_COLOR = "rgba(139, 156, 246, 0.10)";
/** Longest time step used for fading, so a frozen tab does not make everything jump. */
const MAX_FRAME_STEP_MS = 100;

/** Small deterministic pseudo-random number (0–1) for a given integer seed. */
function seededRandom(seed: number): number {
  const value = Math.sin(seed * 12.9898 + 78.233) * 43758.5453;
  return value - Math.floor(value);
}

interface Layout {
  originX: number;
  originY: number;
  cellWidth: number;
  cellHeight: number;
}

/** Scale the fixed 16 × 16 virtual grid to the camera panel without black side margins. */
function computeLayout(width: number, height: number, columns: number, rows: number): Layout {
  return { originX: 0, originY: 0, cellWidth: width / columns, cellHeight: height / rows };
}

/** Faint crosshair and rings, like the axes of a visual-field chart. */
function drawFieldAxes(context: CanvasRenderingContext2D, layout: Layout, columns: number, rows: number): void {
  const halfWidth = (layout.cellWidth * columns) / 2;
  const halfHeight = (layout.cellHeight * rows) / 2;
  const centerX = layout.originX + halfWidth;
  const centerY = layout.originY + halfHeight;

  context.strokeStyle = AXIS_COLOR;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(centerX - halfWidth, centerY);
  context.lineTo(centerX + halfWidth, centerY);
  context.moveTo(centerX, centerY - halfHeight);
  context.lineTo(centerX, centerY + halfHeight);
  context.stroke();

  for (const fraction of [1 / 3, 2 / 3, 1]) {
    context.beginPath();
    context.arc(centerX, centerY, Math.min(halfWidth, halfHeight) * fraction, 0, Math.PI * 2);
    context.stroke();
  }
}

function drawPhosphene(context: CanvasRenderingContext2D, centerX: number, centerY: number, radius: number, intensity: number): void {
  const alpha = Math.min(1, intensity);
  const glow = context.createRadialGradient(centerX, centerY, 0, centerX, centerY, radius);
  glow.addColorStop(0, `rgba(255, 255, 255, ${alpha})`);
  glow.addColorStop(0.22, `rgba(205, 246, 255, ${alpha * 0.8})`);
  glow.addColorStop(0.55, `rgba(139, 156, 246, ${alpha * 0.2})`);
  glow.addColorStop(1, "rgba(139, 156, 246, 0)");
  context.fillStyle = glow;
  context.fillRect(centerX - radius, centerY - radius, radius * 2, radius * 2);
}

/** A few very dim random specks — a hint of the visual noise reported in phosphene studies. */
function drawNoise(context: CanvasRenderingContext2D, width: number, height: number): void {
  context.fillStyle = "rgba(205, 246, 255, 0.06)";
  for (let speck = 0; speck < NOISE_SPECKS_PER_FRAME; speck++) {
    context.fillRect(Math.random() * width, Math.random() * height, 1.5, 1.5);
  }
}

/** Rises quickly toward a brighter target and fades slowly toward a dimmer one. */
function stepIntensity(current: number, target: number, deltaMs: number): number {
  if (target >= current) return current + (target - current) * Math.min(1, deltaMs / RISE_TIME_MS);
  return Math.max(target, current * Math.exp(-deltaMs / PERSISTENCE_MS));
}

export function PhospheneRenderer() {
  const { snapshotRef, viewSettings } = usePipeline();
  const displayedRef = useRef<Float32Array>(new Float32Array(0));
  const lastFrameTimeRef = useRef<number | null>(null);

  const canvasRef = useAnimatedCanvas((context, width, height, now) => {
    const frame = snapshotRef.current.phosphenes;
    const cellCount = frame.columns * frame.rows;
    // A new matrix size (another camera) starts from dark instead of mixing up cells.
    if (displayedRef.current.length !== cellCount) displayedRef.current = new Float32Array(cellCount);

    const deltaMs = Math.min(MAX_FRAME_STEP_MS, now - (lastFrameTimeRef.current ?? now));
    lastFrameTimeRef.current = now;

    context.globalCompositeOperation = "source-over";
    context.fillStyle = "#000000";
    context.fillRect(0, 0, width, height);

    const layout = computeLayout(width, height, frame.columns, frame.rows);
    drawFieldAxes(context, layout, frame.columns, frame.rows);

    const isAnimated = !viewSettings.isReducedMotion;
    const radius = Math.min(layout.cellWidth, layout.cellHeight) * GLOW_RADIUS;
    context.globalCompositeOperation = "lighter"; // overlapping glows add up like light

    for (let row = 0; row < frame.rows; row++) {
      for (let col = 0; col < frame.columns; col++) {
        const index = row * frame.columns + col;
        const intensity = stepIntensity(displayedRef.current[index], frame.activations[row][col], deltaMs);
        displayedRef.current[index] = intensity;
        if (intensity < MIN_VISIBLE_INTENSITY) continue;

        // Real phosphenes don't sit on a perfect grid, so each one gets a fixed small offset.
        const jitterX = (seededRandom(index * 2 + 1) - 0.5) * 2 * POSITION_JITTER * layout.cellWidth;
        const jitterY = (seededRandom(index * 2 + 2) - 0.5) * 2 * POSITION_JITTER * layout.cellHeight;
        const centerX = layout.originX + (col + 0.5) * layout.cellWidth + jitterX;
        const centerY = layout.originY + (row + 0.5) * layout.cellHeight + jitterY;

        const flicker = isAnimated ? 1 - FLICKER_DEPTH + FLICKER_DEPTH * Math.sin(now * FLICKER_SPEED + index * 1.7) : 1;
        drawPhosphene(context, centerX, centerY, radius, intensity * flicker);
      }
    }

    if (isAnimated) drawNoise(context, width, height);
    context.globalCompositeOperation = "source-over";
  });

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Fosfene simulate: puncte de lumină pe conturul persoanelor și obiectelor detectate."
      className="block h-full w-full"
    />
  );
}

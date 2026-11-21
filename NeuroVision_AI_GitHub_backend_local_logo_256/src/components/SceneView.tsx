import { useRef } from "react";
import { usePipeline } from "../context/PipelineContext";
import { useAnimatedCanvas } from "../hooks/useAnimatedCanvas";
import { translateLabel } from "../services/objectDetectionService";
import { describeRelativeDepth } from "../services/depthEstimationService";
import type { PixelOutline } from "../services/objectOutline";
import type { DetectedObject } from "../types/vision";

/*
 * SceneView — the "what the camera sees" window.
 * Shows the live webcam frame with the real outline of every detected object
 * (from its segmentation mask) and a name tag. No boxes: the outline is the same
 * one the phosphenes are built from.
 */

const LABEL_COLOR = "#FFFFFF";
/** Outline colour: the white-cyan of the phosphenes, with a soft glow. */
const OUTLINE_RGB = [205, 246, 255] as const;
const OUTLINE_GLOW_COLOR = "rgba(103, 220, 255, 0.9)";
const OUTLINE_GLOW_BLUR_PX = 8;
const SECTOR_LINE_COLOR = "rgba(255, 255, 255, 0.28)";
const SECTOR_NAMES = ["stânga", "centru", "dreapta"];
const DEPTH_PREVIEW_WIDTH_FRACTION = 0.3;

interface ContentRect {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** Fits a source image inside the canvas without stretching it (like CSS object-fit: contain). */
function fitContain(sourceWidth: number, sourceHeight: number, targetWidth: number, targetHeight: number): ContentRect {
  const scale = Math.min(targetWidth / sourceWidth, targetHeight / sourceHeight);
  const width = sourceWidth * scale;
  const height = sourceHeight * scale;
  return { x: (targetWidth - width) / 2, y: (targetHeight - height) / 2, width, height };
}

/** Draws an image, optionally flipped horizontally (selfie view). */
function drawImageMaybeMirrored(context: CanvasRenderingContext2D, image: CanvasImageSource, rect: ContentRect, isMirrored: boolean): void {
  context.save();
  if (isMirrored) {
    context.translate(rect.x * 2 + rect.width, 0);
    context.scale(-1, 1);
  }
  context.drawImage(image, rect.x, rect.y, rect.width, rect.height);
  context.restore();
}

function drawSectorGuides(context: CanvasRenderingContext2D, rect: ContentRect): void {
  context.save();
  context.strokeStyle = SECTOR_LINE_COLOR;
  context.setLineDash([2, 6]);
  context.lineWidth = 1;
  for (const fraction of [1 / 3, 2 / 3]) {
    const lineX = rect.x + rect.width * fraction;
    context.beginPath();
    context.moveTo(lineX, rect.y);
    context.lineTo(lineX, rect.y + rect.height);
    context.stroke();
  }
  context.setLineDash([]);
  context.font = "600 12px 'Atkinson Hyperlegible Next Variable', sans-serif";
  context.fillStyle = "rgba(255, 255, 255, 0.6)";
  context.textAlign = "center";
  SECTOR_NAMES.forEach((name, index) => {
    context.fillText(name, rect.x + rect.width * (index / 3 + 1 / 6), rect.y + rect.height - 10);
  });
  context.restore();
}

function buildLabelText(detection: DetectedObject): string {
  const base = `${translateLabel(detection.label)} ${Math.round(detection.confidence * 100)}%`;
  if (detection.relativeDepth === null) return base;
  return `${base}, ${describeRelativeDepth(detection.relativeDepth)}`;
}

/** Paints every outline pixel into one transparent layer, at the mask's own resolution. */
function paintOutlineLayer(layer: HTMLCanvasElement, outlines: ReadonlyMap<string, PixelOutline>): void {
  const first = outlines.values().next().value;
  if (!first) {
    layer.width = 1;
    layer.height = 1;
    return;
  }
  layer.width = first.width;
  layer.height = first.height;
  const context = layer.getContext("2d");
  if (!context) return;
  const image = context.createImageData(first.width, first.height);
  const paint = (x: number, y: number) => {
    if (x >= first.width || y >= first.height) return;
    const offset = (y * first.width + x) * 4;
    image.data[offset] = OUTLINE_RGB[0];
    image.data[offset + 1] = OUTLINE_RGB[1];
    image.data[offset + 2] = OUTLINE_RGB[2];
    image.data[offset + 3] = 255;
  };
  for (const outline of outlines.values()) {
    for (const pixelIndex of outline.pixels) {
      const x = pixelIndex % first.width;
      const y = Math.floor(pixelIndex / first.width);
      // 2×2 pixels per outline point, so the line stays visible after scaling.
      paint(x, y);
      paint(x + 1, y);
      paint(x, y + 1);
      paint(x + 1, y + 1);
    }
  }
  context.putImageData(image, 0, 0);
}

function drawOutlineLayer(context: CanvasRenderingContext2D, layer: HTMLCanvasElement, rect: ContentRect, isMirrored: boolean): void {
  if (layer.width <= 1) return;
  context.save();
  context.shadowColor = OUTLINE_GLOW_COLOR;
  context.shadowBlur = OUTLINE_GLOW_BLUR_PX;
  drawImageMaybeMirrored(context, layer, rect, isMirrored);
  context.restore();
}

/** A small name tag at the top-left corner of the object. */
function drawLabel(context: CanvasRenderingContext2D, detection: DetectedObject, rect: ContentRect): void {
  const x = rect.x + detection.bbox.x * rect.width;
  const y = rect.y + detection.bbox.y * rect.height;

  const text = buildLabelText(detection);
  context.font = "700 12px 'Atkinson Hyperlegible Next Variable', sans-serif";
  const textWidth = context.measureText(text).width;
  const labelY = Math.max(rect.y, y - 20);
  context.fillStyle = LABEL_COLOR;
  context.fillRect(x - 1, labelY, textWidth + 12, 19);
  context.fillStyle = "#000000";
  context.textAlign = "left";
  context.fillText(text, x + 5, labelY + 14);
}

function drawDepthPreview(context: CanvasRenderingContext2D, depthCanvas: HTMLCanvasElement, rect: ContentRect, isMirrored: boolean): void {
  if (depthCanvas.width <= 1) return;
  const width = rect.width * DEPTH_PREVIEW_WIDTH_FRACTION;
  const height = (depthCanvas.height / depthCanvas.width) * width;
  const previewRect = { x: rect.x + rect.width - width - 10, y: rect.y + 10, width, height };
  drawImageMaybeMirrored(context, depthCanvas, previewRect, isMirrored);
  context.strokeStyle = "rgba(255,255,255,0.7)";
  context.lineWidth = 1;
  context.strokeRect(previewRect.x, previewRect.y, previewRect.width, previewRect.height);
  context.font = "600 11px 'Atkinson Hyperlegible Next Variable', sans-serif";
  context.fillStyle = "#FFFFFF";
  context.textAlign = "right";
  context.fillText("adâncime relativă (alb = aproape)", previewRect.x + previewRect.width, previewRect.y + previewRect.height + 14);
}

export function SceneView({ label }: { label: string }) {
  const { snapshotRef, videoElement, depthPreviewCanvas, viewSettings, cameraStatus, isDepthOn, depthStatus } = usePipeline();
  // The outline layer is repainted only when a new snapshot brings new outlines.
  const outlineLayerRef = useRef<{ canvas: HTMLCanvasElement; source: ReadonlyMap<string, PixelOutline> | null } | null>(null);

  // The draw function is refreshed on every render, so it always sees current settings.
  const canvasRef = useAnimatedCanvas((context, width, height) => {
    const snapshot = snapshotRef.current;
    context.fillStyle = "#000000";
    context.fillRect(0, 0, width, height);

    if (cameraStatus !== "live" || videoElement.videoWidth === 0) return; // no image yet; the React overlay explains why

    const rect = fitContain(videoElement.videoWidth, videoElement.videoHeight, width, height);
    drawImageMaybeMirrored(context, videoElement, rect, viewSettings.isCameraMirrored);
    if (isDepthOn && depthStatus === "ready") drawDepthPreview(context, depthPreviewCanvas, rect, viewSettings.isCameraMirrored);

    drawSectorGuides(context, rect);

    outlineLayerRef.current ??= { canvas: document.createElement("canvas"), source: null };
    const layer = outlineLayerRef.current;
    if (layer.source !== snapshot.outlines) {
      paintOutlineLayer(layer.canvas, snapshot.outlines);
      layer.source = snapshot.outlines;
    }
    drawOutlineLayer(context, layer.canvas, rect, viewSettings.isCameraMirrored);
    for (const detection of snapshot.detections) drawLabel(context, detection, rect);
  });

  return <canvas ref={canvasRef} role="img" aria-label={label} className="block h-full w-full" />;
}

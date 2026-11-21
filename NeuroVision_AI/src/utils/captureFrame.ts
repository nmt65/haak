const CAPTURE_WIDTH = 448;
const JPEG_QUALITY = 0.8;

/**
 * Captures the current camera frame as a JPEG data URL, for YOLO26n on the backend.
 * The image is mirrored the same way as on screen, so "left" in the result and
 * "left" in the picture agree. Returns null before the camera delivers frames.
 */
export function captureFrameDataUrl(videoElement: HTMLVideoElement, isMirrored: boolean): string | null {
  if (videoElement.videoWidth === 0) return null;
  const canvas = document.createElement("canvas");
  const context = canvas.getContext("2d");
  if (!context) return null;

  canvas.width = CAPTURE_WIDTH;
  canvas.height = Math.round((videoElement.videoHeight / videoElement.videoWidth) * CAPTURE_WIDTH);
  if (isMirrored) {
    context.translate(canvas.width, 0);
    context.scale(-1, 1);
  }
  context.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL("image/jpeg", JPEG_QUALITY);
}

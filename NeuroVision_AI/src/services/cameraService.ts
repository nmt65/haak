/*
 * Webcam access through MediaDevices.getUserMedia.
 * Frames stay in the browser; nothing is uploaded anywhere.
 */

const PREFERRED_WIDTH = 640;
const PREFERRED_HEIGHT = 480;

export class CameraError extends Error {
  constructor(message: string, readonly reason: string) {
    super(message);
    this.name = "CameraError";
  }
}

/** Translates browser camera errors into clear Romanian messages for the user. */
export function describeCameraError(error: unknown): CameraError {
  if (error instanceof CameraError) return error;

  const errorName = error instanceof DOMException || error instanceof Error ? error.name : "UnknownError";
  switch (errorName) {
    case "NotAllowedError":
    case "SecurityError":
      return new CameraError("Accesul la cameră a fost refuzat. Permite camera din setările browserului sau folosește modul demo.", errorName);
    case "NotFoundError":
    case "OverconstrainedError":
      return new CameraError("Nu a fost găsită nicio cameră compatibilă pe acest dispozitiv.", errorName);
    case "NotReadableError":
    case "AbortError":
      return new CameraError("Camera este folosită de altă aplicație sau nu poate fi pornită.", errorName);
    default:
      return new CameraError("Camera nu a putut fi pornită din cauza unei erori neașteptate.", errorName);
  }
}

export interface CameraDevice {
  deviceId: string;
  label: string;
}

/**
 * Lists the video inputs. Browsers only reveal camera names after the user
 * has granted permission once, so call this after a successful start.
 */
export async function listCameras(): Promise<CameraDevice[]> {
  if (!navigator.mediaDevices?.enumerateDevices) return [];
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices
    .filter((device) => device.kind === "videoinput")
    .map((device, index) => ({ deviceId: device.deviceId, label: device.label || `Camera ${index + 1}` }));
}

/**
 * Asks the browser for a camera stream. Throws a CameraError with a readable message.
 * Without a deviceId the browser picks its default camera.
 */
export async function startCameraStream(deviceId?: string): Promise<MediaStream> {
  if (!window.isSecureContext) {
    throw new CameraError("Camera funcționează doar pe HTTPS sau pe localhost.", "InsecureContext");
  }
  if (!navigator.mediaDevices?.getUserMedia) {
    throw new CameraError("Acest browser nu oferă acces la cameră (getUserMedia indisponibil).", "Unsupported");
  }

  try {
    return await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        width: { ideal: PREFERRED_WIDTH },
        height: { ideal: PREFERRED_HEIGHT },
        ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
      },
    });
  } catch (error) {
    throw describeCameraError(error);
  }
}

/** Stops every track so the camera light turns off and the device is released. */
export function stopCameraStream(stream: MediaStream | null): void {
  if (!stream) return;
  for (const track of stream.getTracks()) {
    track.stop();
  }
}

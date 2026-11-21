import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { usePipeline } from "./PipelineContext";
import type { BackendStatus, WorldState } from "../types/backend";
import { derivePacketEvents } from "../utils/perceptualPacket";
import { captureFrameDataUrl } from "../utils/captureFrame";

/*
 * Connection to the optional Python backend (camera + Wi-Fi CSI fusion).
 * The browser sends its current detections; the backend answers ~5×/s with a world state.
 * The dashboard works fully without the backend — this only adds the radio channel.
 */

export const BACKEND_URL = (import.meta.env.VITE_BACKEND_URL as string | undefined) ?? "http://localhost:8000";
const WEBSOCKET_URL = BACKEND_URL.replace(/^http/, "ws") + "/ws/live";
const RECONNECT_DELAY_MS = 3000;
const CAMERA_SEND_INTERVAL_MS = 200;
/** JPEG frames for YOLO26n are sent less often than detections: they are larger and slower to process. */
const FRAME_SEND_INTERVAL_MS = 400;

export type ConnectionState = "off" | "connecting" | "open" | "error";
/** Which detector feeds the fusion: YOLOE already running in the browser, or YOLO26n on the backend. */
export type FusionDetector = "browser" | "yolo";

function useBackendState() {
  const pipeline = usePipeline();
  const { snapshotRef, playRadioCue } = pipeline;
  const [isEnabled, setIsEnabled] = useState(false);
  const [detector, setDetector] = useState<FusionDetector>("browser");
  const detectorRef = useRef(detector);
  detectorRef.current = detector;
  // Latest pipeline values for frame capture inside the timer (avoids restarting the connection).
  const pipelineRef = useRef(pipeline);
  pipelineRef.current = pipeline;
  const [connection, setConnection] = useState<ConnectionState>("off");
  const [world, setWorld] = useState<WorldState | null>(null);
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const previousWorldRef = useRef<WorldState | null>(null);

  const refreshStatus = useCallback(async () => {
    try {
      const response = await fetch(`${BACKEND_URL}/api/status`);
      setStatus((await response.json()) as BackendStatus);
    } catch (error) {
      console.warn("Starea backend-ului nu a putut fi citită:", error);
      setStatus(null);
    }
  }, []);

  /** New world state: keep it, and play a cue when the radio alone starts sensing someone. */
  const handleWorld = useCallback(
    (next: WorldState) => {
      const newEvents = derivePacketEvents(previousWorldRef.current, next);
      previousWorldRef.current = next;
      setWorld(next);
      if (newEvents.some((event) => event.kind === "radio_on" && next.fusion.agreement === "radio_present_camera_none")) {
        playRadioCue();
      }
    },
    [playRadioCue],
  );

  useEffect(() => {
    if (!isEnabled) {
      setConnection("off");
      return;
    }
    let socket: WebSocket | null = null;
    let reconnectTimer = 0;
    let sendTimer = 0;
    let isClosedByUs = false;

    const connect = () => {
      setConnection("connecting");
      socket = new WebSocket(WEBSOCKET_URL);
      socket.onopen = () => {
        setConnection("open");
        void refreshStatus();
      };
      socket.onmessage = (message) => handleWorld(JSON.parse(message.data as string) as WorldState);
      socket.onerror = () => setConnection("error");
      socket.onclose = () => {
        if (isClosedByUs) return;
        setConnection("error");
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
      };
    };

    // Browser detector: send the browser's detections so the backend can fuse them with the radio data.
    const sendCameraDetections = () => {
      if (detectorRef.current !== "browser" || socket?.readyState !== WebSocket.OPEN) return;
      if (pipelineRef.current.cameraStatus !== "live") return;
      socket.send(JSON.stringify({ type: "camera", source: "camera", timestamp: Date.now(), detections: snapshotRef.current.detections }));
    };

    // YOLO detector: send the frame itself; the backend runs YOLO26n on it.
    const sendFrame = () => {
      if (detectorRef.current !== "yolo" || socket?.readyState !== WebSocket.OPEN) return;
      const current = pipelineRef.current;
      if (current.cameraStatus !== "live") return;
      const frame = captureFrameDataUrl(current.videoElement, current.viewSettings.isCameraMirrored);
      if (frame) socket.send(JSON.stringify({ type: "frame", source: "camera", timestamp: Date.now(), frame }));
    };

    connect();
    sendTimer = window.setInterval(sendCameraDetections, CAMERA_SEND_INTERVAL_MS);
    const frameTimer = window.setInterval(sendFrame, FRAME_SEND_INTERVAL_MS);
    return () => {
      isClosedByUs = true;
      window.clearTimeout(reconnectTimer);
      window.clearInterval(sendTimer);
      window.clearInterval(frameTimer);
      socket?.close();
    };
  }, [handleWorld, isEnabled, refreshStatus, snapshotRef]);

  const setCsiMode = useCallback(
    async (mode: string, port?: string): Promise<string | null> => {
      const response = await fetch(`${BACKEND_URL}/api/csi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, port: port || null }),
      });
      await refreshStatus();
      if (response.ok) return null;
      const body = (await response.json()) as { detail?: string };
      return body.detail ?? "Modul CSI nu a putut fi schimbat.";
    },
    [refreshStatus],
  );

  return { isEnabled, setIsEnabled, detector, setDetector, connection, world, status, refreshStatus, setCsiMode };
}

export type BackendContextValue = ReturnType<typeof useBackendState>;

const BackendContext = createContext<BackendContextValue | null>(null);

export function BackendProvider({ children }: { children: ReactNode }) {
  const value = useBackendState();
  return <BackendContext.Provider value={value}>{children}</BackendContext.Provider>;
}

export function useBackend(): BackendContextValue {
  const context = useContext(BackendContext);
  if (!context) throw new Error("useBackend must be used inside <BackendProvider>");
  return context;
}

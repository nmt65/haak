import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import { usePipeline } from "./PipelineContext";
import type { BackendStatus, WorldState } from "../types/backend";
import { derivePacketEvents } from "../utils/perceptualPacket";
import { captureFrameDataUrl } from "../utils/captureFrame";
import { backendConfigError, backendWebSocketUrl, normalizeBackendUrl } from "../utils/backendUrl";

/*
 * Optional Python fusion backend. The main browser detector and 256-channel virtual
 * phosphene encoder work independently when the backend is absent.
 */

const STORAGE_KEY = "neurovision.backend.url";
const RECONNECT_DELAY_MS = 3000;
const CAMERA_SEND_INTERVAL_MS = 200;
const FRAME_SEND_INTERVAL_MS = 400;

export type ConnectionState = "off" | "connecting" | "open" | "error";
export type FusionDetector = "browser" | "yolo";

function defaultBackendUrl(): string {
  const fromBuild = (import.meta.env.VITE_BACKEND_URL as string | undefined)?.trim();
  // On Vercel, localhost points to the visitor's computer, not the deployed API.
  const fallback = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1"
    ? "http://127.0.0.1:8000"
    : "";
  const saved = window.localStorage.getItem(STORAGE_KEY);
  try { return normalizeBackendUrl(saved || fromBuild || fallback); } catch { return ""; }
}

function useBackendState() {
  const pipeline = usePipeline();
  const { snapshotRef, playRadioCue } = pipeline;
  const [backendUrl, setBackendUrlState] = useState(defaultBackendUrl);
  const [isEnabled, setIsEnabled] = useState(false);
  const [detector, setDetector] = useState<FusionDetector>("browser");
  const detectorRef = useRef(detector);
  detectorRef.current = detector;
  const pipelineRef = useRef(pipeline);
  pipelineRef.current = pipeline;
  const [connection, setConnection] = useState<ConnectionState>("off");
  const [connectionError, setConnectionError] = useState<string | null>(null);
  const [world, setWorld] = useState<WorldState | null>(null);
  const [status, setStatus] = useState<BackendStatus | null>(null);
  const previousWorldRef = useRef<WorldState | null>(null);

  const setBackendUrl = useCallback((input: string) => {
    const normalized = normalizeBackendUrl(input);
    setBackendUrlState(normalized);
    setConnectionError(null);
    setStatus(null);
    setWorld(null);
    previousWorldRef.current = null;
    if (normalized) window.localStorage.setItem(STORAGE_KEY, normalized);
    else window.localStorage.removeItem(STORAGE_KEY);
  }, []);

  const refreshStatus = useCallback(async () => {
    if (!backendUrl) {
      setStatus(null);
      return;
    }
    try {
      const response = await fetch(`${backendUrl}/api/status`, { signal: AbortSignal.timeout(6000) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      setStatus((await response.json()) as BackendStatus);
    } catch (error) {
      setStatus(null);
      const detail = error instanceof Error ? error.message : String(error);
      setConnectionError((current) => current || `Nu se poate citi /api/status (${detail}). Verifică serverul, HTTPS și NV_CORS_ORIGINS.`);
    }
  }, [backendUrl]);

  const handleWorld = useCallback((next: WorldState) => {
    const events = derivePacketEvents(previousWorldRef.current, next);
    previousWorldRef.current = next;
    setWorld(next);
    if (events.some((event) => event.kind === "radio_on" && next.fusion.agreement === "radio_present_camera_none")) {
      playRadioCue();
    }
  }, [playRadioCue]);

  useEffect(() => {
    if (!isEnabled) {
      setConnection("off");
      setConnectionError(null);
      return;
    }
    const configError = backendConfigError(backendUrl, window.location.protocol);
    if (configError) {
      setConnection("error");
      setConnectionError(configError);
      return;
    }
    let socket: WebSocket | null = null;
    let reconnectTimer: number | undefined;
    let stopped = false;

    const connect = () => {
      if (stopped) return;
      setConnection("connecting");
      try {
        socket = new WebSocket(backendWebSocketUrl(backendUrl));
      } catch (error) {
        setConnection("error");
        setConnectionError(error instanceof Error ? error.message : "URL WebSocket invalid.");
        return;
      }
      socket.onopen = () => {
        setConnection("open");
        setConnectionError(null);
        void refreshStatus();
      };
      socket.onmessage = (message) => {
        try {
          const next = JSON.parse(message.data as string) as WorldState;
          if (next.type === "world_state") handleWorld(next);
        } catch {
          setConnectionError("Backend-ul a trimis un mesaj WebSocket invalid.");
        }
      };
      socket.onerror = () => {
        setConnection("error");
        setConnectionError("Conexiunea WebSocket a eșuat. Verifică /ws/live, HTTPS/WSS și adresa serverului.");
      };
      socket.onclose = (event) => {
        if (stopped) return;
        setConnection("error");
        if (event.code === 1008) setConnectionError("Originea nu este permisă. Adaugă domeniul frontend-ului în NV_CORS_ORIGINS.");
        reconnectTimer = window.setTimeout(connect, RECONNECT_DELAY_MS);
      };
    };

    const sendCameraDetections = () => {
      if (detectorRef.current !== "browser" || socket?.readyState !== WebSocket.OPEN) return;
      if (pipelineRef.current.cameraStatus !== "live" || socket.bufferedAmount > 1_000_000) return;
      socket.send(JSON.stringify({ type: "camera", source: "camera", timestamp: Date.now(), detections: snapshotRef.current.detections }));
    };

    const sendFrame = () => {
      if (detectorRef.current !== "yolo" || socket?.readyState !== WebSocket.OPEN) return;
      const current = pipelineRef.current;
      if (current.cameraStatus !== "live" || socket.bufferedAmount > 1_000_000) return;
      const frame = captureFrameDataUrl(current.videoElement, current.viewSettings.isCameraMirrored);
      if (frame) socket.send(JSON.stringify({ type: "frame", source: "camera", timestamp: Date.now(), frame }));
    };

    setWorld(null);
    setStatus(null);
    previousWorldRef.current = null;
    setConnectionError(null);
    connect();
    const detectionTimer = window.setInterval(sendCameraDetections, CAMERA_SEND_INTERVAL_MS);
    const frameTimer = window.setInterval(sendFrame, FRAME_SEND_INTERVAL_MS);
    return () => {
      stopped = true;
      window.clearTimeout(reconnectTimer);
      window.clearInterval(detectionTimer);
      window.clearInterval(frameTimer);
      socket?.close();
    };
  }, [backendUrl, handleWorld, isEnabled, refreshStatus, snapshotRef]);

  const setCsiMode = useCallback(async (mode: string, port?: string): Promise<string | null> => {
    if (!backendUrl) return "Adresa backend-ului nu este configurată.";
    try {
      const response = await fetch(`${backendUrl}/api/csi`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode, port: port || null }),
        signal: AbortSignal.timeout(8000),
      });
      if (response.ok) {
        await refreshStatus();
        return null;
      }
      const body = (await response.json()) as { detail?: string };
      return body.detail ?? `Modul CSI nu a putut fi schimbat (HTTP ${response.status}).`;
    } catch (error) {
      return `Backend indisponibil: ${error instanceof Error ? error.message : String(error)}`;
    }
  }, [backendUrl, refreshStatus]);

  return { backendUrl, setBackendUrl, isEnabled, setIsEnabled, detector, setDetector, connection,
    connectionError, world, status, refreshStatus, setCsiMode };
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

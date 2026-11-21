import { createContext, useCallback, useContext, useEffect, useRef, useState, type ReactNode } from "react";
import type { DetectedObject, ModelStatus } from "../types/vision";
import { computeSectorIntensities, type SectorIntensities } from "../services/sectorService";
import { buildContourMatrix, createEmptyPhospheneFrame, toPhospheneFrame, type PhospheneFrame } from "../services/contourEncoder";
import { SpatialAudioEngine } from "../services/spatialAudioService";
import { describeCameraError, listCameras, startCameraStream, stopCameraStream, type CameraDevice } from "../services/cameraService";
import { assignStableIds, isBackgroundSurface, loadObjectDetector, toDetectedObjects, type ObjectDetector } from "../services/objectDetectionService";
import { attachRelativeDepth, loadDepthEstimator, type DepthEstimator, type DepthMap } from "../services/depthEstimationService";
import { buildObjectOutlines, type PixelOutline } from "../services/objectOutline";
import { RateMeter, smoothValue } from "../utils/performance";
import { PHOSPHENE_MATRIX_SIZE } from "../services/phospheneChannels";

/*
 * PipelineContext — the single place where data flows through the system:
 *
 *   camera → objects + pixel masks (+ optional relative depth) → outlines → camera overlay + phosphenes
 *                                                               ↘ sector intensities → direction map, spatial audio
 *
 * Every panel reads the SAME snapshot, so all views always agree.
 */

export type CameraStatus = "off" | "starting" | "live" | "error";

export interface PipelineSnapshot {
  detections: DetectedObject[];
  phosphenes: PhospheneFrame;
  /** Outline of each detection, keyed by detection id: drawn on the camera image and turned into phosphenes. */
  outlines: ReadonlyMap<string, PixelOutline>;
  sectorIntensities: SectorIntensities;
  timestamp: number;
}

export interface PipelineMetrics {
  pipelineFps: number;
  /** Real model inference time. Null until the first detection. */
  inferenceLatencyMs: number | null;
  /** Real depth model time. Null when depth estimation is off. */
  depthLatencyMs: number | null;
}

const EMPTY_METRICS: PipelineMetrics = { pipelineFps: 0, inferenceLatencyMs: null, depthLatencyMs: null };

const MAX_CONSECUTIVE_INFERENCE_FAILURES = 5;
/** Selfie view: the image is shown mirrored, like looking in a mirror. */
const IS_CAMERA_MIRRORED = true;
/** Detections below this confidence (0–1) are ignored. Lower than usual: the model knows 4585 kinds of things. */
const MIN_CONFIDENCE = 0.35;

/** Minimum time between two detection runs. A run takes ~180 ms on WebGPU, so this is rarely the limit. */
const INFERENCE_INTERVAL_MS = 100;
/** Relative depth: frame width for the depth model, how often it runs, and how long a depth map stays usable. */
const DEPTH_WIDTH = 266;
const DEPTH_INTERVAL_MS = 800;
const MAX_DEPTH_AGE_MS = 2500;
/** Exactly 256 virtual channels, arranged in a 16 × 16 matrix. */
const DEFAULT_MATRIX_SIZE = PHOSPHENE_MATRIX_SIZE;

function prefersReducedMotion(): boolean {
  return typeof window !== "undefined" && window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
}

/** Fixed settings the views need to draw the same way as the pipeline computes. */
const VIEW_SETTINGS = {
  isCameraMirrored: IS_CAMERA_MIRRORED,
  isReducedMotion: prefersReducedMotion(),
};

function createEmptySnapshot(): PipelineSnapshot {
  return {
    detections: [],
    phosphenes: createEmptyPhospheneFrame(DEFAULT_MATRIX_SIZE),
    outlines: new Map(),
    sectorIntensities: { left: 0, center: 0, right: 0 },
    timestamp: 0,
  };
}

/** Draws a grayscale depth map into a canvas so it can be shown as a picture-in-picture preview. */
function paintDepthPreview(canvas: HTMLCanvasElement, depthMap: DepthMap): void {
  canvas.width = depthMap.width;
  canvas.height = depthMap.height;
  const context = canvas.getContext("2d");
  if (!context) return;
  const image = context.createImageData(depthMap.width, depthMap.height);
  for (let pixel = 0; pixel < depthMap.data.length; pixel++) {
    const nearness = depthMap.data[pixel];
    image.data[pixel * 4] = nearness;
    image.data[pixel * 4 + 1] = nearness;
    image.data[pixel * 4 + 2] = nearness;
    image.data[pixel * 4 + 3] = 255;
  }
  context.putImageData(image, 0, 0);
}

function usePipelineState() {
  const [snapshot, setSnapshot] = useState<PipelineSnapshot>(createEmptySnapshot);
  const [metrics, setMetrics] = useState<PipelineMetrics>(EMPTY_METRICS);

  const [cameraStatus, setCameraStatusState] = useState<CameraStatus>("off");
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [cameraDevices, setCameraDevices] = useState<CameraDevice[]>([]);
  const [selectedCameraId, setSelectedCameraId] = useState<string | null>(null);
  const [modelStatus, setModelStatus] = useState<ModelStatus>("idle");
  const [modelError, setModelError] = useState<string | null>(null);

  const [isDepthOn, setIsDepthOn] = useState(false);
  const [depthStatus, setDepthStatus] = useState<ModelStatus>("idle");
  const [depthError, setDepthError] = useState<string | null>(null);
  const [depthProgress, setDepthProgress] = useState(0);

  const [isAudioOn, setIsAudioOn] = useState(false);
  const [audioError, setAudioError] = useState<string | null>(null);

  // Refs give the animation loop the latest values without restarting it.
  const isDepthOnRef = useRef(isDepthOn);
  const cameraStatusRef = useRef<CameraStatus>("off");
  isDepthOnRef.current = isDepthOn;

  const snapshotRef = useRef(snapshot);
  const rateMeterRef = useRef(new RateMeter());
  const latencyRef = useRef<number | null>(null);
  const depthLatencyRef = useRef<number | null>(null);

  const audioEngineRef = useRef<SpatialAudioEngine | null>(null);
  if (!audioEngineRef.current) audioEngineRef.current = new SpatialAudioEngine();
  const audioEngine = audioEngineRef.current;

  const detectorRef = useRef<ObjectDetector | null>(null);
  const depthEstimatorRef = useRef<DepthEstimator | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  // Created once (lazy useState initializer) and kept for the app's lifetime.
  const [videoElement] = useState(() => document.createElement("video"));
  const [depthInputCanvas] = useState(() => document.createElement("canvas"));
  const [depthPreviewCanvas] = useState(() => document.createElement("canvas"));

  const isInferringRef = useRef(false);
  const isEstimatingDepthRef = useRef(false);
  const lastInferenceStartRef = useRef(0);
  const lastDepthStartRef = useRef(0);
  const latestDepthRef = useRef<{ map: DepthMap; capturedAt: number } | null>(null);
  const inferenceFailuresRef = useRef(0);
  const nextTrackIdRef = useRef(1);

  const setCameraStatus = useCallback((status: CameraStatus) => {
    cameraStatusRef.current = status;
    setCameraStatusState(status);
  }, []);

  /** Clears all pipeline output (used when the camera starts or stops) so no stale data is shown. */
  const resetPipeline = useCallback(() => {
    rateMeterRef.current.reset();
    latencyRef.current = null;
    depthLatencyRef.current = null;
    latestDepthRef.current = null;
    const emptySnapshot = createEmptySnapshot();
    snapshotRef.current = emptySnapshot; // canvases read the ref directly
    setSnapshot(emptySnapshot);
    setMetrics(EMPTY_METRICS);
  }, []);

  /** Runs one full pass of the pipeline on a set of detections and publishes the result. */
  const processDetections = useCallback(
    (detections: DetectedObject[], outlines: ReadonlyMap<string, PixelOutline>, inferenceMs: number) => {
      // Phosphenes follow each object's own outline (see services/objectOutline.ts and contourEncoder.ts).
      const matrixSize = PHOSPHENE_MATRIX_SIZE;
      const phosphenes = toPhospheneFrame(buildContourMatrix(detections, outlines, matrixSize, IS_CAMERA_MIRRORED));

      const sectorIntensities = computeSectorIntensities(detections);
      audioEngine.update(sectorIntensities);
      latencyRef.current = smoothValue(latencyRef.current, inferenceMs);

      const nextSnapshot: PipelineSnapshot = { detections, phosphenes, outlines, sectorIntensities, timestamp: Date.now() };
      snapshotRef.current = nextSnapshot;
      setSnapshot(nextSnapshot);
      setMetrics({
        pipelineFps: rateMeterRef.current.tick(),
        inferenceLatencyMs: latencyRef.current,
        depthLatencyMs: isDepthOnRef.current ? depthLatencyRef.current : null,
      });
    },
    [audioEngine],
  );

  /** Copies the current video frame into a canvas of the given width (keeps the aspect ratio). */
  const captureFrame = useCallback(
    (canvas: HTMLCanvasElement, width: number) => {
      canvas.width = width;
      canvas.height = Math.round((videoElement.videoHeight / videoElement.videoWidth) * width);
      canvas.getContext("2d")?.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
    },
    [videoElement],
  );

  const isVideoReady = useCallback(() => videoElement.readyState >= 2 && videoElement.videoWidth > 0, [videoElement]);

  /** Runs one real inference if the previous one finished and enough time passed. */
  const runCameraInference = useCallback(
    (now: number) => {
      const detector = detectorRef.current;
      const isDue = now - lastInferenceStartRef.current >= INFERENCE_INTERVAL_MS;
      if (!detector || !isVideoReady() || isInferringRef.current || !isDue) return;
      if (cameraStatusRef.current !== "live") return;

      const frameWidth = videoElement.videoWidth;
      const frameHeight = videoElement.videoHeight;
      isInferringRef.current = true;
      lastInferenceStartRef.current = now;
      const inferenceStart = performance.now();

      detector
        .detect(videoElement, frameWidth, frameHeight)
        .then((rawDetections) => {
          const inferenceMs = performance.now() - inferenceStart;
          inferenceFailuresRef.current = 0;
          if (cameraStatusRef.current !== "live") return;

          const confident = rawDetections.filter((raw) => raw.score >= MIN_CONFIDENCE && !isBackgroundSurface(raw, frameWidth, frameHeight));
          const detections = toDetectedObjects(confident, frameWidth, frameHeight, IS_CAMERA_MIRRORED);

          const latestDepth = latestDepthRef.current;
          const isDepthFresh = latestDepth !== null && performance.now() - latestDepth.capturedAt < MAX_DEPTH_AGE_MS;
          const depthMap = isDepthOnRef.current && isDepthFresh ? latestDepth.map : null;
          const withDepth = attachRelativeDepth(detections, depthMap, IS_CAMERA_MIRRORED);

          // These steps keep the order of the detections, so index i still belongs to confident[i].
          const trackedDetections = assignStableIds(snapshotRef.current.detections, withDepth, () => `obj-${nextTrackIdRef.current++}`);
          const masks = new Map(trackedDetections.map((detection, index) => [detection.id, confident[index].mask]));
          processDetections(trackedDetections, buildObjectOutlines(masks), inferenceMs);
        })
        .catch((error: unknown) => {
          inferenceFailuresRef.current += 1;
          console.error("Inferența a eșuat:", error);
          if (inferenceFailuresRef.current >= MAX_CONSECUTIVE_INFERENCE_FAILURES) {
            detectorRef.current = null;
            setModelStatus("error");
            setModelError("Detecția a eșuat de mai multe ori la rând. Reîncearcă modelul.");
          }
        })
        .finally(() => {
          isInferringRef.current = false;
        });
    },
    [isVideoReady, processDetections, videoElement],
  );

  /** With depth on: runs the depth model at a lower rate than detection. */
  const runDepthEstimation = useCallback(
    (now: number) => {
      const estimator = depthEstimatorRef.current;
      const isDue = now - lastDepthStartRef.current >= DEPTH_INTERVAL_MS;
      if (!isDepthOnRef.current || !estimator || !isVideoReady() || isEstimatingDepthRef.current || !isDue) return;
      if (cameraStatusRef.current !== "live") return;

      captureFrame(depthInputCanvas, DEPTH_WIDTH);
      isEstimatingDepthRef.current = true;
      lastDepthStartRef.current = now;
      const depthStart = performance.now();

      estimator
        .estimate(depthInputCanvas)
        .then((depthMap) => {
          depthLatencyRef.current = smoothValue(depthLatencyRef.current, performance.now() - depthStart, 0.3);
          latestDepthRef.current = { map: depthMap, capturedAt: performance.now() };
          paintDepthPreview(depthPreviewCanvas, depthMap);
        })
        .catch((error: unknown) => {
          console.error("Estimarea adâncimii a eșuat:", error);
          depthEstimatorRef.current = null;
          setDepthStatus("error");
          setDepthError("Estimarea adâncimii a eșuat pe acest dispozitiv. Detecția continuă fără adâncime.");
        })
        .finally(() => {
          isEstimatingDepthRef.current = false;
        });
    },
    [captureFrame, depthInputCanvas, depthPreviewCanvas, isVideoReady],
  );

  // One animation loop drives detection and depth; each runs only when it is due.
  useEffect(() => {
    let frameId = 0;
    const tick = (now: number) => {
      frameId = requestAnimationFrame(tick);
      runCameraInference(now);
      runDepthEstimation(now);
    };
    frameId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frameId);
  }, [runCameraInference, runDepthEstimation]);

  // Release camera and audio when the app unmounts.
  useEffect(() => {
    return () => {
      stopCameraStream(streamRef.current);
      void audioEngine.disable();
    };
  }, [audioEngine]);

  const loadModel = useCallback(async () => {
    setModelStatus("loading");
    setModelError(null);
    try {
      detectorRef.current = await loadObjectDetector();
      inferenceFailuresRef.current = 0;
      setModelStatus("ready");
    } catch (error) {
      console.error("Modelul nu a putut fi încărcat:", error);
      setModelStatus("error");
      setModelError("Modelul de detecție nu a putut porni în acest browser. Reîncearcă sau folosește Chrome/Edge actualizat.");
    }
  }, []);

  const loadDepthModel = useCallback(async () => {
    setDepthStatus("loading");
    setDepthError(null);
    setDepthProgress(0);
    try {
      depthEstimatorRef.current = await loadDepthEstimator((percent) => setDepthProgress(percent));
      setDepthStatus("ready");
    } catch (error) {
      console.error("Modelul de adâncime nu a putut fi încărcat:", error);
      setDepthStatus("error");
      setDepthError("Modelul de adâncime nu s-a putut descărca sau porni. Verifică conexiunea și reîncearcă.");
    }
  }, []);

  /** Turns relative depth on/off. The model is downloaded on first use. */
  const setDepthEnabled = useCallback(
    (isOn: boolean) => {
      setIsDepthOn(isOn);
      isDepthOnRef.current = isOn;
      latestDepthRef.current = null;
      if (isOn && !depthEstimatorRef.current) void loadDepthModel();
    },
    [loadDepthModel],
  );

  const releaseCamera = useCallback(() => {
    stopCameraStream(streamRef.current);
    streamRef.current = null;
    videoElement.srcObject = null;
  }, [videoElement]);

  /** Starts (or restarts) the camera. Without a deviceId it reuses the last chosen camera. */
  const startCameraDevice = useCallback(
    async (deviceId?: string) => {
      resetPipeline();
      setCameraError(null);
      setCameraStatus("starting");

      // Start downloading the model while the user answers the permission prompt.
      if (!detectorRef.current) void loadModel();

      try {
        releaseCamera();
        const stream = await startCameraStream(deviceId ?? selectedCameraId ?? undefined);
        streamRef.current = stream;
        videoElement.muted = true;
        videoElement.playsInline = true;
        videoElement.srcObject = stream;
        await videoElement.play();
        setCameraStatus("live");

        const activeDeviceId = stream.getVideoTracks()[0]?.getSettings().deviceId ?? null;
        setSelectedCameraId(activeDeviceId);
        setCameraDevices(await listCameras());
      } catch (error) {
        releaseCamera();
        setCameraError(describeCameraError(error).message);
        setCameraStatus("error");
      }
    },
    [loadModel, releaseCamera, resetPipeline, selectedCameraId, setCameraStatus, videoElement],
  );

  // Separate wrappers so a button's click event is never mistaken for a deviceId.
  const startCamera = useCallback(() => startCameraDevice(), [startCameraDevice]);
  const selectCamera = useCallback((deviceId: string) => startCameraDevice(deviceId), [startCameraDevice]);

  const stopCamera = useCallback(() => {
    releaseCamera();
    setCameraStatus("off");
    resetPipeline();
  }, [releaseCamera, resetPipeline, setCameraStatus]);

  const enableAudio = useCallback(async () => {
    if (!SpatialAudioEngine.isSupported()) {
      setAudioError("Browserul nu suportă Web Audio API, așa că sunetul nu poate fi pornit.");
      return;
    }
    try {
      await audioEngine.enable();
      setAudioError(null);
      setIsAudioOn(true);
    } catch (error) {
      console.error("Audio indisponibil:", error);
      setAudioError("Browserul a blocat sunetul. Apasă din nou pe „Sunet”.");
      setIsAudioOn(false);
    }
  }, [audioEngine]);

  const disableAudio = useCallback(async () => {
    setIsAudioOn(false);
    await audioEngine.disable();
  }, [audioEngine]);

  const playRadioCue = useCallback(() => audioEngine.playRadioCue(), [audioEngine]);

  const isCameraProcessing = cameraStatus === "live" && modelStatus === "ready";

  return {
    snapshot,
    snapshotRef,
    metrics,
    videoElement,
    depthPreviewCanvas,
    viewSettings: VIEW_SETTINGS,
    cameraStatus,
    cameraError,
    cameraDevices,
    selectedCameraId,
    modelStatus,
    modelError,
    isCameraProcessing,
    startCamera,
    selectCamera,
    stopCamera,
    retryModel: loadModel,
    isDepthOn,
    depthStatus,
    depthError,
    depthProgress,
    setDepthEnabled,
    retryDepthModel: loadDepthModel,
    isAudioOn,
    audioError,
    enableAudio,
    disableAudio,
    playRadioCue,
  };
}

export type PipelineContextValue = ReturnType<typeof usePipelineState>;

const PipelineContext = createContext<PipelineContextValue | null>(null);

export function PipelineProvider({ children }: { children: ReactNode }) {
  // The value changes on every pipeline update (~8×/s); all panels are meant to re-render with it.
  const value = usePipelineState();
  return <PipelineContext.Provider value={value}>{children}</PipelineContext.Provider>;
}

/** Access the shared pipeline state from any component inside <PipelineProvider>. */
export function usePipeline(): PipelineContextValue {
  const context = useContext(PipelineContext);
  if (!context) throw new Error("usePipeline must be used inside <PipelineProvider>");
  return context;
}

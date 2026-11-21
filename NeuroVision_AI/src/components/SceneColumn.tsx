import { Camera, Loader2, RefreshCw } from "lucide-react";
import { usePipeline } from "../context/PipelineContext";
import { translateLabel } from "../services/objectDetectionService";
import { describeRelativeDepth } from "../services/depthEstimationService";
import { SceneView } from "./SceneView";
import { Button, Section } from "./ui";
import type { HorizontalSector } from "../types/vision";

const SECTOR_WORDS: Record<HorizontalSector, string> = { left: "stânga", center: "centru", right: "dreapta" };

/**
 * Message over the camera window when there is no image or the model is not ready.
 * "Pornește" also turns sound on: it is one click, and browsers only allow audio after a click.
 */
function WindowOverlay() {
  const { cameraStatus, cameraError, modelStatus, modelError, startCamera, retryModel, isAudioOn, enableAudio } = usePipeline();

  const start = () => {
    void startCamera();
    if (!isAudioOn) void enableAudio();
  };

  if (cameraStatus === "starting") {
    return (
      <div className="absolute inset-0 grid place-items-center text-sm text-paper">
        <span className="flex items-center gap-2">
          <Loader2 size={16} className="animate-spin" aria-hidden /> Se pornește camera
        </span>
      </div>
    );
  }

  if (cameraStatus === "error" || cameraStatus === "off") {
    const isError = cameraStatus === "error";
    return (
      <div className="absolute inset-0 grid place-items-center p-6 text-center">
        <div className="max-w-xs">
          {isError && (
            <p className="mb-4 text-base text-paper" role="alert">
              {cameraError}
            </p>
          )}
          <Button variant="primary" size="lg" icon={isError ? RefreshCw : Camera} onClick={start}>
            {isError ? "Încearcă din nou" : "Pornește"}
          </Button>
        </div>
      </div>
    );
  }

  if (modelStatus === "loading") {
    return (
      <p className="absolute inset-x-3 bottom-3 flex items-center gap-2 rounded-lg bg-black/75 px-3 py-2 text-sm text-paper">
        <Loader2 size={14} className="animate-spin" aria-hidden /> Se descarcă modelul de detecție.
      </p>
    );
  }
  if (modelStatus === "error") {
    return (
      <div className="absolute inset-x-3 bottom-3 rounded-lg bg-black/85 px-3 py-3 text-sm text-paper" role="alert">
        <p>{modelError}</p>
        <Button size="sm" icon={RefreshCw} onClick={() => void retryModel()} className="mt-2">
          Reîncearcă modelul
        </Button>
      </div>
    );
  }
  return null;
}

/** One line per detected object: what, how sure, which side, relative depth. */
function DetectionList() {
  const { snapshot, cameraStatus } = usePipeline();
  if (cameraStatus !== "live") return null;
  if (snapshot.detections.length === 0) {
    return <p className="text-sm text-graphite">Nicio detecție. O zonă fără detecții nu este neapărat liberă.</p>;
  }
  return (
    <ul className="grid gap-1 text-sm">
      {snapshot.detections.map((detection) => (
        <li key={detection.id} className="flex justify-between gap-3 border-t border-rule pt-1">
          <span className="font-bold text-ink">{translateLabel(detection.label)}</span>
          <span className="text-graphite">
            {SECTOR_WORDS[detection.horizontalSector]}, {Math.round(detection.confidence * 100)}%
            {detection.relativeDepth !== null && `, ${describeRelativeDepth(detection.relativeDepth)}`}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function SceneColumn() {
  return (
    <Section title="Camera">
      <div className="optical-window aspect-[4/3]">
        <SceneView label="Imagine live de la cameră, cu obiectele detectate" />
        <WindowOverlay />
      </div>
      <div className="mt-3">
        <DetectionList />
      </div>
    </Section>
  );
}

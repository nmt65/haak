import { useId } from "react";
import { Camera, CameraOff, Layers, Volume2, VolumeX } from "lucide-react";
import { usePipeline } from "../context/PipelineContext";
import { Button, StatusLight } from "./ui";

/** Uploaded NeuroVision AI mark. Transparent PNG keeps the gradient legible on the light UI. */
function Logo() {
  return (
    <a href="/" className="flex min-h-11 shrink-0 items-center gap-2 rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-cobalt" aria-label="NeuroVision AI — pagina principală">
      <img
        src="/branding/neurovision-icon.png"
        alt=""
        aria-hidden="true"
        width="70"
        height="46"
        className="h-10 w-[66px] object-contain sm:h-11 sm:w-[73px]"
      />
      <span className="whitespace-nowrap text-lg font-semibold tracking-[-0.045em] text-ink sm:text-[22px]">
        NeuroVision<span className="font-medium">Ai</span>
      </span>
    </a>
  );
}

/** Plain-language processing status, derived only from real state. */
function ProcessingStatus() {
  const { cameraStatus, modelStatus, isCameraProcessing, metrics } = usePipeline();
  if (isCameraProcessing) {
    return <StatusLight tone="live">Live, {metrics.pipelineFps.toFixed(1).replace(".", ",")} cadre/s</StatusLight>;
  }
  if (cameraStatus === "error") return <StatusLight tone="error">Camera nu pornește</StatusLight>;
  if (cameraStatus === "starting") return <StatusLight tone="busy">Pornește camera</StatusLight>;
  if (cameraStatus === "off") return <StatusLight tone="off">Camera e oprită</StatusLight>;
  if (modelStatus === "loading") return <StatusLight tone="busy">Se descarcă modelul</StatusLight>;
  return <StatusLight tone="error">Modelul nu e disponibil</StatusLight>;
}

/** Camera picker, shown only when the device has more than one camera. */
function CameraPicker() {
  const { cameraDevices, selectedCameraId, selectCamera } = usePipeline();
  const selectId = useId();
  if (cameraDevices.length < 2) return null;
  return (
    <>
      <label htmlFor={selectId} className="sr-only">
        Alege camera
      </label>
      <select
        id={selectId}
        value={selectedCameraId ?? ""}
        onChange={(event) => void selectCamera(event.target.value)}
        className="h-8 max-w-[180px] rounded-lg border border-rule bg-raised px-2 text-sm text-ink"
      >
        {cameraDevices.map((device) => (
          <option key={device.deviceId} value={device.deviceId}>
            {device.label}
          </option>
        ))}
      </select>
    </>
  );
}

function DepthButton() {
  const { isDepthOn, setDepthEnabled, depthStatus, depthProgress } = usePipeline();
  let label = isDepthOn ? "Adâncime pornită" : "Adâncime";
  if (isDepthOn && depthStatus === "loading") label = `Adâncime ${Math.round(depthProgress)}%`;
  return (
    <Button size="sm" variant={isDepthOn ? "primary" : "secondary"} icon={Layers} aria-pressed={isDepthOn} onClick={() => setDepthEnabled(!isDepthOn)}>
      {label}
    </Button>
  );
}

function SoundButton() {
  const { isAudioOn, enableAudio, disableAudio } = usePipeline();
  return (
    <Button
      size="sm"
      variant={isAudioOn ? "primary" : "secondary"}
      icon={isAudioOn ? Volume2 : VolumeX}
      aria-pressed={isAudioOn}
      onClick={() => void (isAudioOn ? disableAudio() : enableAudio())}
    >
      Sunet
    </Button>
  );
}

function CameraButton() {
  const { cameraStatus, startCamera, stopCamera } = usePipeline();
  const isRunning = cameraStatus === "live" || cameraStatus === "starting";
  return isRunning ? (
    <Button size="sm" icon={CameraOff} onClick={stopCamera}>
      Oprește camera
    </Button>
  ) : (
    <Button size="sm" variant="primary" icon={Camera} onClick={() => void startCamera()}>
      Pornește camera
    </Button>
  );
}

export function Header() {
  return (
    <header className="sticky top-0 z-30 border-b border-rule bg-chassis/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1540px] flex-wrap items-center gap-3 px-4 py-3 sm:px-6">
        <Logo />
        <ProcessingStatus />
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <CameraPicker />
          <CameraButton />
          <SoundButton />
          <DepthButton />
        </div>
      </div>
    </header>
  );
}

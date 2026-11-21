import { useEffect, useState } from "react";
import { Plug, PlugZap } from "lucide-react";
import { useBackend } from "../context/BackendContext";
import { CsiSignalPlot } from "./fusion/CsiSignalPlot";
import { Button, Note, Readout, Section, SegmentedControl, StatusLight } from "./ui";
import type { Agreement, ModelOutput } from "../types/backend";

/*
 * Wi-Fi CSI fusion: the Python backend reads radio data, runs the presence
 * classifiers and fuses them with the camera detections.
 */

const AGREEMENT_TEXT: Record<Agreement, string> = {
  no_data: "Fără date de la niciun senzor.",
  radio_only: "Doar radio: camera nu trimite cadre recente.",
  camera_only: "Doar camera: nu există date radio.",
  agree_present: "Camera și radioul indică amândouă prezență umană.",
  agree_absent: "Camera și radioul nu indică prezență umană.",
  camera_person_radio_absent: "Dezacord: camera vede o persoană, radioul nu indică prezență.",
  radio_present_camera_none: "Dezacord: radioul indică prezență, camera nu vede nicio persoană (poate fi în afara câmpului).",
};

const CSI_MODES = [
  { value: "simulated", label: "Simulat" },
  { value: "recorded", label: "Înregistrat" },
  { value: "hardware", label: "ESP32" },
];

const MODEL_LABELS = { csi: "CSI", camera: "Cameră", fusion: "Fuziune" } as const;
const STATUS_REFRESH_MS = 3000;

function ConnectionLight() {
  const { connection, backendUrl } = useBackend();
  if (connection === "open") return <StatusLight tone="ok">Conectat la {backendUrl}</StatusLight>;
  if (connection === "connecting") return <StatusLight tone="busy">Se conectează</StatusLight>;
  if (connection === "error") return <StatusLight tone="error">Backend-ul nu răspunde; se reîncearcă</StatusLight>;
  return <StatusLight tone="off">Deconectat</StatusLight>;
}

/** CSI source, serial port and which detector feeds the fusion. */
function SourceControls() {
  const { status, setCsiMode, detector, setDetector } = useBackend();
  const [port, setPort] = useState("COM3");
  const [modeError, setModeError] = useState<string | null>(null);
  if (!status) return null;

  const changeMode = async (mode: string) => {
    setModeError(await setCsiMode(mode, mode === "hardware" ? port : undefined));
  };

  return (
    <div className="grid gap-3">
      <div className="flex flex-wrap items-end gap-4">
        <SegmentedControl label="Sursa CSI" options={CSI_MODES} value={status.csi_mode} onChange={(mode) => void changeMode(mode)} />
        <label className="text-sm text-ink">
          Port serial
          <input value={port} onChange={(event) => setPort(event.target.value)} className="ml-2 h-9 w-24 rounded-lg border border-rule bg-raised px-2 text-sm text-ink" />
        </label>
        <SegmentedControl
          label="Detector pentru fuziune"
          options={[
            { value: "browser", label: "YOLOE (browser)" },
            { value: "yolo", label: "YOLO26n (backend)" },
          ]}
          value={detector}
          onChange={setDetector}
        />
      </div>
      {modeError && <p role="alert" className="text-sm text-live">{modeError}</p>}
      {status.yolo_error && <p role="alert" className="text-sm text-live">YOLO26n: {status.yolo_error}</p>}
      {status.runtime.trained_on_synthetic_data && (
        <Note>Modelele CSI încărcate sunt antrenate pe date sintetice: probabilitățile arată doar că software-ul funcționează.</Note>
      )}
    </div>
  );
}

function ModelBar({ label, output, isAvailable }: { label: string; output: ModelOutput | null; isAvailable: boolean }) {
  const percent = output ? Math.round(output.probability * 100) : 0;
  let verdict = "model neîncărcat";
  if (isAvailable) verdict = output ? (output.is_present ? "prezență" : "absență") : "fără date";
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3 text-sm">
        <span className="text-ink">{label}</span>
        <span className="font-mono text-graphite">{output ? `${percent}%, ${verdict}` : verdict}</span>
      </div>
      <div className="mt-1.5 h-1.5 bg-rule" role="meter" aria-label={`${label}: probabilitate prezență`} aria-valuemin={0} aria-valuemax={100} aria-valuenow={percent}>
        <div className={`h-full transition-[width] duration-200 ${output?.is_present ? "bg-cobalt" : "bg-graphite"}`} style={{ width: `${percent}%` }} />
      </div>
    </div>
  );
}

/** Radio signal next to the presence classifiers and the camera–radio agreement. */
function LiveRadio() {
  const { world } = useBackend();
  if (!world) return null;
  const available = new Set(world.runtime.available_models);

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <div>
        <div className="optical-window aspect-[16/7]">
          <CsiSignalPlot world={world} />
        </div>
        {world.radio_error && <p role="alert" className="mt-2 text-sm text-live">{world.radio_error}</p>}
      </div>
      <div className="grid content-start gap-4">
        {(Object.keys(MODEL_LABELS) as (keyof typeof MODEL_LABELS)[]).map((mode) => (
          <ModelBar key={mode} label={MODEL_LABELS[mode]} output={world.models[mode]} isAvailable={available.has(mode)} />
        ))}
        <p className="border-l-2 border-cobalt pl-3 text-sm text-ink">{AGREEMENT_TEXT[world.fusion.agreement]}</p>
      </div>
    </div>
  );
}

export function CsiPanel() {
  const { isEnabled, setIsEnabled, backendUrl, setBackendUrl, connection, connectionError, status, refreshStatus } = useBackend();
  const [urlDraft, setUrlDraft] = useState(backendUrl);
  const [urlError, setUrlError] = useState<string | null>(null);

  useEffect(() => setUrlDraft(backendUrl), [backendUrl]);

  const saveUrl = () => {
    try {
      setBackendUrl(urlDraft);
      setUrlError(null);
    } catch (error) {
      setUrlError(error instanceof Error ? error.message : "URL invalid.");
    }
  };

  // YOLO and model states change while running, so re-read them every few seconds.
  useEffect(() => {
    if (connection !== "open") return;
    const timer = window.setInterval(() => void refreshStatus(), STATUS_REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [connection, refreshStatus]);

  return (
    <Section
      title="Wi-Fi CSI"
      actions={
        <Button variant={isEnabled ? "secondary" : "primary"} size="sm" icon={isEnabled ? Plug : PlugZap} onClick={() => setIsEnabled(!isEnabled)}>
          {isEnabled ? "Deconectează" : "Conectează"}
        </Button>
      }
    >
      <div className="grid gap-5">
        <div className="grid gap-2 rounded-lg border border-rule bg-raised p-3">
          <label htmlFor="backend-url" className="text-sm font-semibold text-ink">Adresa backend-ului Python</label>
          <div className="flex flex-wrap gap-2">
            <input
              id="backend-url"
              type="url"
              value={urlDraft}
              onChange={(event) => setUrlDraft(event.target.value)}
              onKeyDown={(event) => { if (event.key === "Enter") saveUrl(); }}
              placeholder="https://backend-ul-tau.onrender.com"
              className="h-10 min-w-[220px] flex-1 rounded-lg border border-rule bg-chassis px-3 font-mono text-sm text-ink"
              spellCheck={false}
            />
            <Button onClick={saveUrl} size="sm">Salvează URL</Button>
          </div>
          <p className="text-xs text-graphite">Local: http://localhost:8000 · Pe Vercel: URL HTTPS al serverului Python separat.</p>
          {urlError && <p className="text-sm text-live" role="alert">{urlError}</p>}
          {connectionError && <p className="text-sm text-live" role="alert">{connectionError}</p>}
        </div>
        <ConnectionLight />
        {status && connection === "open" && (
          <>
            <dl className="grid grid-cols-2 border-y border-rule sm:grid-cols-4">
              <Readout label="Modele" value={status.runtime.available_models.join(", ") || "niciunul"} />
              <Readout label="Antrenate pe" value={status.runtime.dataset} />
              <Readout label="Sursă CSI" value={status.csi_mode} />
              <Readout label="YOLO26n" value={status.yolo_latency_ms !== null ? `${status.yolo_status}, ${status.yolo_latency_ms} ms` : status.yolo_status} />
            </dl>
            <SourceControls />
            <LiveRadio />
          </>
        )}
      </div>
    </Section>
  );
}

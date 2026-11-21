import { useEffect } from "react";
import { PipelineProvider, usePipeline } from "../context/PipelineContext";
import { BackendProvider, useBackend } from "../context/BackendContext";
import { ErrorBoundary } from "../components/ErrorBoundary";
import { Header } from "../components/Header";
import { LiveAnnouncer } from "../components/LiveAnnouncer";
import { SceneColumn } from "../components/SceneColumn";
import { MapColumn } from "../components/MapColumn";
import { PhospheneRenderer } from "../components/PhospheneRenderer";
import { CsiPanel } from "../components/CsiPanel";
import { Button, Section } from "../components/ui";
import { encodeChannelPacket } from "../services/phospheneChannels";

/** Global shortcut: "M" mutes all sound immediately. */
function useMuteShortcut() {
  const { isAudioOn, disableAudio } = usePipeline();
  useEffect(() => {
    const handleKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      const isTyping = target.closest("input, textarea, select") !== null;
      if (isTyping || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key.toLowerCase() === "m" && isAudioOn) void disableAudio();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [isAudioOn, disableAudio]);
}

/**
 * Contour phosphenes. The violet dot is radio (Wi-Fi CSI) presence: radio has no
 * position, so it is shown apart from the matrix instead of inside it.
 */
function PhospheneColumn() {
  const { world } = useBackend();
  const { snapshot } = usePipeline();
  const isRadioPresent = world?.models.csi?.is_present ?? false;
  const packet = encodeChannelPacket(snapshot.phosphenes, snapshot.timestamp);
  const exportChannels = () => {
    const file = new Blob([JSON.stringify(packet, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = `neurovision-256-canale-${Date.now()}.json`;
    link.click();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return (
    <Section title="Fosfene · 256 canale" description="Matrice virtuală 16 × 16, intensități 0–1 (simulare)"
      actions={<Button variant="secondary" size="sm" onClick={exportChannels}>Export JSON</Button>}>
      <div className="mb-3 flex items-baseline justify-between gap-3 rounded-lg border border-rule bg-raised px-3 py-2 text-sm" aria-live="polite">
        <span>Canale active</span>
        <strong className="font-mono tabular-nums text-cobalt">{packet.active_count} / 256</strong>
      </div>
      <div className="optical-window aspect-[4/3]">
        <PhospheneRenderer />
        {world && (
          <span
            className={`absolute bottom-3 right-3 h-4 w-4 rounded-full ${isRadioPresent ? "bg-violet shadow-[0_0_18px_6px_rgba(148,102,255,0.7)]" : "bg-raised"}`}
            role="img"
            aria-label={isRadioPresent ? "Radioul indică prezență" : "Radioul nu indică prezență"}
          />
        )}
      </div>
    </Section>
  );
}

function LiveScreen() {
  useMuteShortcut();
  return (
    <div className="min-h-screen bg-chassis text-ink">
      <Header />
      <main className="mx-auto grid max-w-[1540px] grid-cols-[minmax(0,1fr)] gap-10 px-4 py-6 sm:px-6 sm:py-8" aria-label="NeuroVision AI live">
        <div className="grid gap-8 md:grid-cols-2 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)_minmax(0,1fr)]">
          <ErrorBoundary areaName="Camera">
            <SceneColumn />
          </ErrorBoundary>
          <ErrorBoundary areaName="Fosfene">
            <PhospheneColumn />
          </ErrorBoundary>
          <div className="md:col-span-2 xl:col-span-1">
            <ErrorBoundary areaName="Direcții">
              <MapColumn />
            </ErrorBoundary>
          </div>
        </div>
        <ErrorBoundary areaName="Wi-Fi CSI">
          <CsiPanel />
        </ErrorBoundary>
      </main>
      <footer className="mx-auto max-w-[1540px] px-4 pb-8 text-sm text-graphite sm:px-6">
        Prototip de cercetare, nu dispozitiv medical: nu stimulează creierul și nu garantează un drum sigur.
      </footer>
      <LiveAnnouncer />
    </div>
  );
}

export function App() {
  return (
    <PipelineProvider>
      <BackendProvider>
        <LiveScreen />
      </BackendProvider>
    </PipelineProvider>
  );
}

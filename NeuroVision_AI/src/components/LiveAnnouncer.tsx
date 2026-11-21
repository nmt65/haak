import { useEffect, useRef, useState } from "react";
import { usePipeline } from "../context/PipelineContext";
import type { HorizontalSector } from "../types/vision";

/*
 * Screen-reader announcements of which sectors contain observations.
 * To avoid overwhelming the user it only speaks when the set of sectors
 * changes AND stays the same for a moment, and never more often than every few seconds.
 */

const STABLE_FOR_MS = 800;
const MIN_GAP_BETWEEN_ANNOUNCEMENTS_MS = 4000;
const ANNOUNCE_THRESHOLD = 0.25;

const SECTOR_WORDS: Record<HorizontalSector, string> = { left: "stânga", center: "centru", right: "dreapta" };

function describeSectors(sectors: HorizontalSector[]): string {
  if (sectors.length === 0) return "Nicio observație în acest moment.";
  return `Observații în: ${sectors.map((sector) => SECTOR_WORDS[sector]).join(", ")}.`;
}

export function LiveAnnouncer() {
  const { snapshot } = usePipeline();
  const [message, setMessage] = useState("");
  const candidateRef = useRef({ key: "", since: 0 });
  const lastAnnouncedRef = useRef({ key: "", at: 0 });

  const activeSectors = (["left", "center", "right"] as HorizontalSector[]).filter((sector) => snapshot.sectorIntensities[sector] >= ANNOUNCE_THRESHOLD);
  const sectorKey = activeSectors.join(",");

  useEffect(() => {
    const now = performance.now();
    if (candidateRef.current.key !== sectorKey) {
      candidateRef.current = { key: sectorKey, since: now };
      return;
    }
    const isStable = now - candidateRef.current.since >= STABLE_FOR_MS;
    const isNew = sectorKey !== lastAnnouncedRef.current.key;
    const isGapRespected = now - lastAnnouncedRef.current.at >= MIN_GAP_BETWEEN_ANNOUNCEMENTS_MS;
    if (isStable && isNew && isGapRespected) {
      lastAnnouncedRef.current = { key: sectorKey, at: now };
      setMessage(describeSectors(activeSectors));
    }
  });

  return (
    <div className="sr-only" role="status" aria-live="polite">
      {message}
    </div>
  );
}

import { usePipeline } from "../context/PipelineContext";
import { DirectionalMap } from "./DirectionalMap";
import { Section } from "./ui";
import type { HorizontalSector } from "../types/vision";

const SECTOR_LABELS: Record<HorizontalSector, string> = { left: "Stânga", center: "Centru", right: "Dreapta" };
const SECTOR_ORDER: HorizontalSector[] = ["left", "center", "right"];

/** How strongly each side is occupied, as three thin gauges. */
function SectorGauges() {
  const { snapshot } = usePipeline();
  return (
    <div className="grid grid-cols-3 gap-4">
      {SECTOR_ORDER.map((sector) => {
        const percent = Math.round(snapshot.sectorIntensities[sector] * 100);
        return (
          <div key={sector}>
            <div className="flex items-baseline justify-between text-sm">
              <span className="text-graphite">{SECTOR_LABELS[sector]}</span>
              <span className="font-bold text-ink">{percent}%</span>
            </div>
            <div
              className="mt-1.5 h-1 bg-rule"
              role="meter"
              aria-label={`Ocupare ${SECTOR_LABELS[sector].toLowerCase()}`}
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={percent}
            >
              <div className="h-full bg-cobalt transition-[width] duration-200" style={{ width: `${percent}%` }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function MapColumn() {
  return (
    <Section title="Direcții" className="w-full max-w-2xl">
      <div className="paper-well aspect-[4/3] px-2 pt-2">
        <DirectionalMap />
      </div>
      <div className="mt-4">
        <SectorGauges />
      </div>
    </Section>
  );
}

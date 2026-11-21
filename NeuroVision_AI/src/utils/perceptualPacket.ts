import type { WorldState } from "../types/backend";

/*
 * The "perceptual packet": a compact digital summary of what changed in the
 * world state. These are abstract software events (like spikes in a raster
 * plot), NOT physical neural stimulation commands.
 */

export type PacketChannel = "left" | "center" | "right" | "radio";

export type PacketEventKind = "object_on" | "object_off" | "radio_on" | "radio_off" | "motion_on";

export interface PacketEvent {
  at: number; // ms timestamp
  channel: PacketChannel;
  kind: PacketEventKind;
  detail: string;
}

const SECTORS = ["left", "center", "right"] as const;

function labelsBySector(world: WorldState | null): Record<(typeof SECTORS)[number], Set<string>> {
  const result = { left: new Set<string>(), center: new Set<string>(), right: new Set<string>() };
  for (const object of world?.objects ?? []) result[object.sector]?.add(object.label);
  return result;
}

/** Compares two consecutive world states and returns the events that happened in between. */
export function derivePacketEvents(previous: WorldState | null, next: WorldState): PacketEvent[] {
  const at = Math.round(next.timestamp * 1000);
  const events: PacketEvent[] = [];

  const before = labelsBySector(previous);
  const after = labelsBySector(next);
  for (const sector of SECTORS) {
    for (const label of after[sector]) {
      if (!before[sector].has(label)) events.push({ at, channel: sector, kind: "object_on", detail: label });
    }
    for (const label of before[sector]) {
      if (!after[sector].has(label)) events.push({ at, channel: sector, kind: "object_off", detail: label });
    }
  }

  const wasRadioPresent = previous?.models.csi?.is_present ?? false;
  const isRadioPresent = next.models.csi?.is_present ?? false;
  if (isRadioPresent && !wasRadioPresent) events.push({ at, channel: "radio", kind: "radio_on", detail: "prezență radio" });
  if (!isRadioPresent && wasRadioPresent) events.push({ at, channel: "radio", kind: "radio_off", detail: "fără prezență radio" });

  const wasMoving = previous?.radio?.motion_detected ?? false;
  if (next.radio?.motion_detected && !wasMoving) events.push({ at, channel: "radio", kind: "motion_on", detail: "mișcare în canal" });

  return events;
}

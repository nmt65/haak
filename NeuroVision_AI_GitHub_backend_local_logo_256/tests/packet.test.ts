import { describe, expect, it } from "vitest";
import { derivePacketEvents } from "../src/utils/perceptualPacket";
import type { WorldState } from "../src/types/backend";

/** Minimal world state with the given objects and radio presence. */
function world(objects: { label: string; sector: "left" | "center" | "right" }[], isRadioPresent: boolean, isMoving = false): WorldState {
  return {
    type: "world_state",
    timestamp: 1000,
    objects: objects.map((object) => ({ ...object, confidence: 0.9, relative_depth: null })),
    unknown_sectors: [],
    camera: { is_fresh: true, age_s: 0, person_count: 0, source: "camera", detector: "COCO-SSD" },
    radio: {
      mode: "simulated",
      is_simulated: true,
      description: "",
      sample_id: null,
      ground_truth_presence: null,
      motion_energy: 0,
      motion_detected: isMoving,
      plot: [],
      position_information: "niciuna",
    },
    radio_error: null,
    models: { csi: { probability: isRadioPresent ? 0.9 : 0.1, is_present: isRadioPresent, latency_ms: 0.2 }, camera: null, fusion: null },
    fusion: { is_synchronised: true, agreement: "no_data" },
    runtime: { run_dir: "", dataset: "SYNTHETIC", trained_on_synthetic_data: true, available_models: ["csi"] },
  };
}

describe("perceptual packet events", () => {
  it("reports objects appearing and disappearing per sector", () => {
    const events = derivePacketEvents(world([{ label: "chair", sector: "left" }], false), world([{ label: "person", sector: "center" }], false));
    expect(events.map((event) => `${event.kind}:${event.channel}:${event.detail}`).sort()).toEqual([
      "object_off:left:chair",
      "object_on:center:person",
    ]);
  });

  it("puts radio presence on its own channel, never on a direction", () => {
    const events = derivePacketEvents(world([], false), world([], true, true));
    expect(events.every((event) => event.channel === "radio")).toBe(true);
    expect(events.map((event) => event.kind).sort()).toEqual(["motion_on", "radio_on"]);
  });

  it("produces no events when nothing changes", () => {
    const state = world([{ label: "chair", sector: "left" }], true);
    expect(derivePacketEvents(state, state)).toEqual([]);
  });
});

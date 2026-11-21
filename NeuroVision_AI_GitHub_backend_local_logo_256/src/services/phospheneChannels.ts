import type { PhospheneFrame } from "./contourEncoder";

/** Virtual simulation channels, not intracranial electrodes or a stimulation protocol. */
export const PHOSPHENE_COLUMNS = 16;
export const PHOSPHENE_ROWS = 16;
export const PHOSPHENE_CHANNEL_COUNT = PHOSPHENE_COLUMNS * PHOSPHENE_ROWS;
export const PHOSPHENE_MATRIX_SIZE = { columns: PHOSPHENE_COLUMNS, rows: PHOSPHENE_ROWS } as const;

export interface ChannelPacket {
  type: "virtual_phosphene_channels";
  version: 1;
  timestamp: number;
  width: 16;
  height: 16;
  channel_count: 256;
  values: number[];
  active_count: number;
  description: string;
}

/** Fixed row-major channel IDs 0..255; data can be visualized or exported for offline research. */
export function encodeChannelPacket(frame: PhospheneFrame, timestamp: number): ChannelPacket {
  if (frame.columns !== PHOSPHENE_COLUMNS || frame.rows !== PHOSPHENE_ROWS ||
      frame.activations.length !== PHOSPHENE_ROWS || frame.activations.some((row) => row.length !== PHOSPHENE_COLUMNS)) {
    throw new Error("Matricea de fosfene trebuie să aibă exact 16 × 16 celule.");
  }
  const values = frame.activations.flat().map((value) => Number.isFinite(value) ? Math.max(0, Math.min(1, value)) : 0);
  return {
    type: "virtual_phosphene_channels",
    version: 1,
    timestamp,
    width: 16,
    height: 16,
    channel_count: 256,
    values,
    active_count: values.filter((value) => value > 0).length,
    description: "Only simulated activation intensity (0-1); NOT neural stimulation instructions.",
  };
}

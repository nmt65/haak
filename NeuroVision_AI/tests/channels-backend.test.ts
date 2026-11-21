import { describe, expect, it } from "vitest";
import { createEmptyPhospheneFrame } from "../src/services/contourEncoder";
import { encodeChannelPacket, PHOSPHENE_CHANNEL_COUNT, PHOSPHENE_MATRIX_SIZE } from "../src/services/phospheneChannels";
import { backendConfigError, backendWebSocketUrl, normalizeBackendUrl } from "../src/utils/backendUrl";

describe("256-channel virtual phosphene encoder", () => {
  it("always exports 256 fixed row-major channels", () => {
    expect(PHOSPHENE_CHANNEL_COUNT).toBe(256);
    const frame = createEmptyPhospheneFrame(PHOSPHENE_MATRIX_SIZE);
    frame.activations[0][0] = 0.7;
    frame.activations[15][15] = 1;
    const packet = encodeChannelPacket(frame, 123);
    expect(packet).toMatchObject({ width: 16, height: 16, channel_count: 256, active_count: 2, timestamp: 123 });
    expect(packet.values).toHaveLength(256);
    expect(packet.values[0]).toBe(0.7);
    expect(packet.values[255]).toBe(1);
  });

  it("rejects a wrong matrix and sanitizes nonfinite or out of range activation", () => {
    expect(() => encodeChannelPacket(createEmptyPhospheneFrame({ columns: 24, rows: 18 }), 1)).toThrow();
    const frame = createEmptyPhospheneFrame(PHOSPHENE_MATRIX_SIZE);
    frame.activations[0][1] = Number.NaN;
    frame.activations[0][2] = 42;
    frame.activations[0][3] = -1;
    expect(encodeChannelPacket(frame, 1).values.slice(1, 4)).toEqual([0, 1, 0]);
  });
});

describe("backend configuration", () => {
  it("uses secure WebSockets when backend is HTTPS", () => {
    expect(normalizeBackendUrl(" https://fusion.example.org/ ")).toBe("https://fusion.example.org");
    expect(backendWebSocketUrl("https://fusion.example.org")).toBe("wss://fusion.example.org/ws/live");
  });
  it("allows HTTP on localhost and prevents mixed content on production", () => {
    expect(backendWebSocketUrl("http://localhost:8000")).toBe("ws://localhost:8000/ws/live");
    expect(backendConfigError("http://localhost:8000", "https:")).not.toBeNull();
    expect(backendConfigError("http://localhost:8000", "http:")).toBeNull();
    expect(backendConfigError("", "https:")).not.toBeNull();
  });
  it("rejects invalid endpoints", () => {
    expect(() => normalizeBackendUrl("ws://backend.example.org")).toThrow();
    expect(() => normalizeBackendUrl("https://backend.example.org/api/status")).toThrow();
    expect(() => normalizeBackendUrl("https://user:pass@backend.example.org")).toThrow();
  });
});

import { useRef } from "react";
import { useAnimatedCanvas } from "../../hooks/useAnimatedCanvas";
import type { WorldState } from "../../types/backend";

/*
 * Heat strip of the 12 CSI feature channels over the 3-second window
 * (rows = channels, columns = time). Each row is scaled on its own so that
 * changes are visible; colours show relative change, not absolute amplitude.
 */

const CHANNEL_NAMES = ["medie", "abatere", "q05", "q25", "mediană", "q75", "q95", "rms", "min", "max", "Δ medie", "Δ abatere"];
const LABEL_WIDTH = 64;

function rowRange(values: number[]): [number, number] {
  let low = Infinity;
  let high = -Infinity;
  for (const value of values) {
    low = Math.min(low, value);
    high = Math.max(high, value);
  }
  return [low, high > low ? high : low + 1];
}

export function CsiSignalPlot({ world }: { world: WorldState | null }) {
  const worldRef = useRef(world);
  worldRef.current = world;

  const canvasRef = useAnimatedCanvas((context, width, height) => {
    context.fillStyle = "#000000";
    context.fillRect(0, 0, width, height);
    const plot = worldRef.current?.radio?.plot;
    if (!plot) {
      context.fillStyle = "#94A3B8";
      context.font = "14px 'Atkinson Hyperlegible Next Variable', sans-serif";
      context.fillText("Fără date CSI", 16, 28);
      return;
    }

    const rowHeight = height / plot.length;
    const cellWidth = (width - LABEL_WIDTH) / plot[0].length;
    plot.forEach((row, rowIndex) => {
      const [low, high] = rowRange(row);
      row.forEach((value, columnIndex) => {
        const level = (value - low) / (high - low);
        context.fillStyle = `rgba(139, 156, 246, ${0.08 + level * 0.85})`;
        context.fillRect(LABEL_WIDTH + columnIndex * cellWidth, rowIndex * rowHeight, cellWidth + 0.5, rowHeight - 1);
      });
      context.fillStyle = "#94A3B8";
      context.font = "11px 'Atkinson Hyperlegible Next Variable', sans-serif";
      context.fillText(CHANNEL_NAMES[rowIndex] ?? `c${rowIndex}`, 6, rowIndex * rowHeight + rowHeight * 0.7);
    });
  });

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Semnal CSI: 12 caracteristici ale amplitudinii pe ultimele 3 secunde"
      className="block h-full w-full"
    />
  );
}

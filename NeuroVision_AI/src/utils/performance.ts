/**
 * Measures how many updates happen per second over a sliding time window.
 * Values are real measurements from `performance.now()`, never estimates.
 */
export class RateMeter {
  private timestamps: number[] = [];

  constructor(private readonly windowMs = 1000) {}

  /** Records one update and returns the current rate in updates per second. */
  tick(now: number = performance.now()): number {
    this.timestamps.push(now);
    const windowStart = now - this.windowMs;
    while (this.timestamps.length > 0 && this.timestamps[0] < windowStart) {
      this.timestamps.shift();
    }
    return this.rate();
  }

  rate(): number {
    return (this.timestamps.length * 1000) / this.windowMs;
  }

  reset(): void {
    this.timestamps = [];
  }
}

/** Exponential moving average, used to keep latency readouts from jumping around. */
export function smoothValue(previous: number | null, next: number, weightOfNext = 0.2): number {
  if (previous === null) return next;
  return previous + (next - previous) * weightOfNext;
}

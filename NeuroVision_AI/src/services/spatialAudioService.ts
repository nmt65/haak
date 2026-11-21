import type { HorizontalSector } from "../types/vision";
import type { SectorIntensities } from "./sectorService";

/*
 * Spatial audio engine (Web Audio API).
 * Each occupied sector plays its own tone, left to right: stereo position, pitch and
 * rhythm differ per sector so the direction can be recognised by ear. Loudness follows
 * how prominent the observation is in the image. It is NOT a distance measurement.
 */

interface SectorSound {
  pan: number; // -1 = fully left, +1 = fully right
  frequencyHz: number;
  pulseCount: number;
  waveform: OscillatorType;
}

const SECTOR_SOUNDS: Record<HorizontalSector, SectorSound> = {
  left: { pan: -0.9, frequencyHz: 392, pulseCount: 1, waveform: "sine" },
  center: { pan: 0, frequencyHz: 523, pulseCount: 2, waveform: "triangle" },
  right: { pan: 0.9, frequencyHz: 659, pulseCount: 1, waveform: "sine" },
};

const SECTOR_ORDER: HorizontalSector[] = ["left", "center", "right"];

/** Low triple pulse for "radio presence without visual confirmation". */
const RADIO_SOUND: SectorSound = { pan: 0, frequencyHz: 220, pulseCount: 3, waveform: "sine" };

const PULSE_DURATION_S = 0.12;
const PULSE_GAP_S = 0.08;
const GAP_BETWEEN_SECTORS_S = 0.22;
const ATTACK_S = 0.012;
const SCHEDULE_AHEAD_S = 0.02;

/** A new pattern is announced quickly; an unchanged one is repeated slowly. */
const MIN_INTERVAL_CHANGED_MS = 700;
const REPEAT_INTERVAL_UNCHANGED_MS = 2600;

/** Master volume (0–1): starts low on purpose, sudden loud tones are unpleasant. */
const MASTER_VOLUME = 0.3;
/** A sector plays a tone once its intensity (0–1) reaches this value. */
const TRIGGER_THRESHOLD = 0.32;

export class SpatialAudioEngine {
  private context: AudioContext | null = null;
  private masterGain: GainNode | null = null;
  private lastCueTimeMs = -Infinity;
  private lastPatternKey = "";
  private busyUntilS = 0;

  static isSupported(): boolean {
    return typeof window !== "undefined" && "AudioContext" in window;
  }

  get isRunning(): boolean {
    return this.context !== null && this.context.state === "running";
  }

  /** Must be called from a user gesture (click / key press) because of browser autoplay rules. */
  async enable(): Promise<void> {
    if (!this.context) {
      this.context = new AudioContext();

      // A compressor protects against sudden loud peaks.
      const limiter = this.context.createDynamicsCompressor();
      limiter.threshold.value = -18;
      limiter.ratio.value = 8;

      this.masterGain = this.context.createGain();
      this.masterGain.gain.value = MASTER_VOLUME;
      this.masterGain.connect(limiter);
      limiter.connect(this.context.destination);
    }
    await this.context.resume();
  }

  /** Immediately silences everything and releases the audio device. */
  async disable(): Promise<void> {
    const context = this.context;
    this.context = null;
    this.masterGain = null;
    this.lastPatternKey = "";
    this.busyUntilS = 0;
    if (context) await context.close();
  }

  /** Returns the sectors that should be announced, in playback order (left to right). */
  selectSectors(intensities: SectorIntensities): HorizontalSector[] {
    return SECTOR_ORDER.filter((sector) => intensities[sector] >= TRIGGER_THRESHOLD);
  }

  /**
   * Called on every pipeline update. Plays cues only when enough time has passed
   * (debounce), so the user is never flooded with overlapping sounds.
   */
  update(intensities: SectorIntensities, nowMs: number = performance.now()): void {
    if (!this.isRunning || !this.context) return;

    const sectors = this.selectSectors(intensities);
    if (sectors.length === 0) {
      this.lastPatternKey = "";
      return;
    }

    const patternKey = sectors.join(",");
    const hasPatternChanged = patternKey !== this.lastPatternKey;
    const requiredGapMs = hasPatternChanged ? MIN_INTERVAL_CHANGED_MS : REPEAT_INTERVAL_UNCHANGED_MS;
    const elapsedMs = nowMs - this.lastCueTimeMs;
    const isPreviousSequencePlaying = this.context.currentTime < this.busyUntilS;
    if (elapsedMs < requiredGapMs || isPreviousSequencePlaying) return;

    this.lastPatternKey = patternKey;
    this.lastCueTimeMs = nowMs;
    this.playSequence(sectors, intensities);
  }

  /**
   * Radio-only presence (Wi-Fi CSI): a low, centred, non-directional cue.
   * Centred on purpose, because CSI says nothing about WHERE the person is.
   */
  playRadioCue(): void {
    if (!this.isRunning || !this.context) return;
    const startTime = Math.max(this.context.currentTime + SCHEDULE_AHEAD_S, this.busyUntilS);
    let pulseStart = startTime;
    for (let pulse = 0; pulse < RADIO_SOUND.pulseCount; pulse++) {
      this.schedulePulse(RADIO_SOUND, 0.6, pulseStart);
      pulseStart += PULSE_DURATION_S + PULSE_GAP_S * 2;
    }
    this.busyUntilS = pulseStart;
  }

  private playSequence(sectors: HorizontalSector[], intensities: SectorIntensities): void {
    if (!this.context) return;
    let startTime = this.context.currentTime + SCHEDULE_AHEAD_S;
    for (const sector of sectors) {
      const loudness = 0.45 + intensities[sector] * 0.55;
      startTime = this.scheduleSectorCue(sector, loudness, startTime) + GAP_BETWEEN_SECTORS_S;
    }
    this.busyUntilS = startTime;
  }

  /** Schedules the pulses for one sector and returns the time when they end. */
  private scheduleSectorCue(sector: HorizontalSector, loudness: number, startTime: number): number {
    const sound = SECTOR_SOUNDS[sector];
    let pulseStart = startTime;
    for (let pulse = 0; pulse < sound.pulseCount; pulse++) {
      this.schedulePulse(sound, loudness, pulseStart);
      pulseStart += PULSE_DURATION_S + PULSE_GAP_S;
    }
    return pulseStart;
  }

  /** One short tone: oscillator → envelope → stereo panner → master volume. */
  private schedulePulse(sound: SectorSound, loudness: number, startTime: number): void {
    if (!this.context || !this.masterGain) return;
    const oscillator = this.context.createOscillator();
    oscillator.type = sound.waveform;
    oscillator.frequency.value = sound.frequencyHz;

    // Short attack and release avoid audible clicks at the start and end of each tone.
    const envelope = this.context.createGain();
    envelope.gain.setValueAtTime(0, startTime);
    envelope.gain.linearRampToValueAtTime(loudness, startTime + ATTACK_S);
    envelope.gain.exponentialRampToValueAtTime(0.0001, startTime + PULSE_DURATION_S);

    const panner = this.createPanner(sound.pan);
    oscillator.connect(envelope);
    envelope.connect(panner);
    panner.connect(this.masterGain);

    oscillator.start(startTime);
    oscillator.stop(startTime + PULSE_DURATION_S + 0.02);
    oscillator.onended = () => {
      oscillator.disconnect();
      envelope.disconnect();
      panner.disconnect();
    };
  }

  /** Uses StereoPannerNode when available, otherwise a basic PannerNode fallback. */
  private createPanner(pan: number): AudioNode {
    const context = this.context as AudioContext;
    if (typeof context.createStereoPanner === "function") {
      const stereoPanner = context.createStereoPanner();
      stereoPanner.pan.value = pan;
      return stereoPanner;
    }
    const panner = context.createPanner();
    panner.panningModel = "equalpower";
    panner.positionX.value = pan;
    panner.positionZ.value = 1 - Math.abs(pan);
    return panner;
  }
}

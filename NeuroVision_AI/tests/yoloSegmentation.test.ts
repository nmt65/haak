import { describe, expect, it } from "vitest";
import {
  buildObjectMask,
  computeLetterbox,
  decodeCandidates,
  inputBoxToFrame,
  MASK_COEFFICIENTS,
  removeDuplicateObjects,
  ROW_LENGTH,
  type Candidate,
} from "../src/services/yoloSegmentation";

/** One output0 row: box in input pixels, score, class index, mask coefficients (all zero unless given). */
function makeRow(box: [number, number, number, number], score: number, classIndex: number, firstCoefficient = 0): number[] {
  const row = new Array<number>(ROW_LENGTH).fill(0);
  row.splice(0, 6, ...box, score, classIndex);
  row[6] = firstCoefficient;
  return row;
}

function makeCandidate(box: [number, number, number, number], score: number, classIndex = 0): Candidate {
  const [left, top, right, bottom] = box;
  return { classIndex, score, box: { left, top, right, bottom }, coefficients: new Float32Array(MASK_COEFFICIENTS) };
}

describe("letterbox", () => {
  it("fits a 4:3 frame into the square input with grey bands above and below", () => {
    const letterbox = computeLetterbox(640, 480, 384);
    expect(letterbox.scale).toBeCloseTo(0.6);
    expect(letterbox.padX).toBe(0);
    expect(letterbox.padY).toBe(48); // (384 − 288) / 2
  });

  it("maps boxes back to frame pixels and clips them", () => {
    const letterbox = computeLetterbox(640, 480, 384);
    const box = inputBoxToFrame({ left: -10, top: 48, right: 192, bottom: 336 }, letterbox);
    expect(box).toEqual({ left: 0, top: 0, right: 320, bottom: 480 });
  });
});

describe("decoding", () => {
  it("keeps rows above the score threshold and skips empty rows", () => {
    const output = new Float32Array([...makeRow([10, 20, 110, 220], 0.9, 7), ...makeRow([0, 0, 0, 0], 0, 0), ...makeRow([5, 5, 50, 50], 0.1, 3)]);
    const candidates = decodeCandidates(output, 3, 0.25);
    expect(candidates).toHaveLength(1);
    expect(candidates[0]).toMatchObject({ classIndex: 7, box: { left: 10, top: 20, right: 110, bottom: 220 } });
    expect(candidates[0].score).toBeCloseTo(0.9);
  });

  it("drops the same object reported under a second name", () => {
    const bus = makeCandidate([20, 200, 800, 730], 0.85, 5);
    const sameBusOtherName = makeCandidate([30, 230, 800, 740], 0.75, 1234);
    const person = makeCandidate([50, 390, 230, 900], 0.88, 0);
    const kept = removeDuplicateObjects([bus, sameBusOtherName, person], 0.6, 20);
    expect(kept).toEqual([person, bus]);
  });
});

describe("object mask", () => {
  it("fills only the object's box when the mask value is positive everywhere", () => {
    const protoSize = 4;
    const prototypes = new Float32Array(MASK_COEFFICIENTS * protoSize * protoSize);
    prototypes.fill(1, 0, protoSize * protoSize); // first prototype = +1 everywhere
    const candidate = makeCandidate([0, 0, 192, 192], 0.9);
    candidate.coefficients[0] = 1;

    const letterbox = computeLetterbox(384, 384, 384); // square frame, no grey bands
    const mask = buildObjectMask(candidate, prototypes, protoSize, letterbox, 8);
    expect(mask.width).toBe(8);
    expect(mask.height).toBe(8);
    expect(mask.data[0]).toBe(1); // inside the box (top-left quarter)
    expect(mask.data[3 * 8 + 3]).toBe(1);
    expect(mask.data[5 * 8 + 5]).toBe(0); // outside the box
  });

  it("stays empty when the mask value is negative", () => {
    const protoSize = 4;
    const prototypes = new Float32Array(MASK_COEFFICIENTS * protoSize * protoSize).fill(-1);
    const candidate = makeCandidate([0, 0, 384, 384], 0.9);
    candidate.coefficients[0] = 1;
    const mask = buildObjectMask(candidate, prototypes, protoSize, computeLetterbox(384, 384, 384), 8);
    expect(mask.data.every((value) => value === 0)).toBe(true);
  });
});

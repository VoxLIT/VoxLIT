import { describe, expect, it } from "vitest";
import type { DiarizationSegment } from "../types";
import {
  mostAlikeSpeakers,
  orderSegments,
  similarityMatrix,
  similarityWords,
  uncertainSegments,
} from "../page/similarity";
import { result } from "./fixtures";

const seg = (id: string, start: number, speaker: string): DiarizationSegment => ({
  id,
  start,
  end: start + 1,
  speaker,
  confidence: 0.5,
  confidence_bucket: "high",
});

// Interleaved speakers, so time order and grouped order differ.
const interleaved = [seg("a", 0, "S1"), seg("b", 1, "S0"), seg("c", 2, "S1"), seg("d", 3, "S0"), seg("short", 4, "S0")];
const vectors = { a: [1, 0], b: [0, 1], c: [0.8, 0.6], d: [0.6, 0.8] };

describe("orderSegments", () => {
  it("keeps time order and drops segments with no embedding", () => {
    expect(orderSegments(interleaved, vectors, ["S0", "S1"], false).map((s) => s.id)).toEqual(["a", "b", "c", "d"]);
  });

  it("groups by the run's speaker order, then by start time", () => {
    expect(orderSegments(interleaved, vectors, ["S0", "S1"], true).map((s) => s.id)).toEqual(["b", "d", "a", "c"]);
  });
});

describe("similarityMatrix", () => {
  it("is symmetric with ones on the diagonal", () => {
    const ordered = orderSegments(interleaved, vectors, ["S0", "S1"], false);
    const m = similarityMatrix(ordered, vectors);
    const n = ordered.length;
    for (let i = 0; i < n; i++) {
      expect(m[i * n + i]).toBe(1);
      for (let j = 0; j < n; j++) expect(m[i * n + j]).toBeCloseTo(m[j * n + i]);
    }
    expect(m[0 * n + 2]).toBeCloseTo(0.8); // a·c
  });
});

describe("mostAlikeSpeakers", () => {
  it("returns the cross-speaker pair with the highest mean similarity", () => {
    const ordered = orderSegments(result.segments, result.embeddings, result.speakers, false);
    const pair = mostAlikeSpeakers(ordered, similarityMatrix(ordered, result.embeddings), result.speakers);
    // seg_1·seg_3 = 0, seg_1·seg_4 = 1, seg_2·seg_3 = 0.1, seg_2·seg_4 = 0.9
    expect(pair?.a).toBe("SPEAKER_00");
    expect(pair?.b).toBe("SPEAKER_01");
    expect(pair?.similarity).toBeCloseTo(0.5);
  });

  it("is null with a single speaker", () => {
    const one = [seg("a", 0, "S0"), seg("b", 1, "S0")];
    const ordered = orderSegments(one, vectors, ["S0"], false);
    expect(mostAlikeSpeakers(ordered, similarityMatrix(ordered, vectors), ["S0"])).toBeNull();
  });
});

describe("words and lists", () => {
  it("puts similarity into words", () => {
    expect(similarityWords(0.9)).toBe("sound very alike");
    expect(similarityWords(0.4)).toBe("sound somewhat alike");
    expect(similarityWords(0.1)).toBe("sound different");
  });

  it("lists uncertain segments only, never unscored ones", () => {
    expect(uncertainSegments(result.segments).map((s) => s.id)).toEqual(["seg_4"]);
  });
});

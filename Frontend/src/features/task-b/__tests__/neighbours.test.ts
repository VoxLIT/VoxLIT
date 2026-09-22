import { describe, expect, it } from "vitest";
import { cosine, nearestByCosine } from "../page/neighbours";
import { formatClock } from "../page/segmentWords";
import { result } from "./fixtures";

describe("nearestByCosine", () => {
  it("ranks by cosine in embedding space, not by closeness on the 2D map", () => {
    // On the map seg_2 is seg_1's neighbour; in the model's space it is seg_4.
    const ids = nearestByCosine(result.embeddings, "seg_1").map((n) => n.id);
    expect(ids[0]).toBe("seg_4");
    expect(ids).toEqual(["seg_4", "seg_2", "seg_3"]);
  });

  it("never lists the segment itself", () => {
    expect(nearestByCosine(result.embeddings, "seg_1").map((n) => n.id)).not.toContain("seg_1");
  });

  it("is empty for a segment too short to embed", () => {
    expect(nearestByCosine(result.embeddings, "seg_5")).toEqual([]);
  });

  it("caps the list at k", () => {
    expect(nearestByCosine(result.embeddings, "seg_1", 2)).toHaveLength(2);
  });

  it("does not assume unit-length vectors", () => {
    expect(cosine([2, 0], [5, 0])).toBeCloseTo(1);
    expect(cosine([0, 0], [1, 0])).toBe(0);
  });
});

describe("formatClock", () => {
  it("formats minutes and tenths, rolling 59.96 s over to a full minute", () => {
    expect(formatClock(72.4)).toBe("01:12.4");
    expect(formatClock(59.96)).toBe("01:00.0");
    expect(formatClock(0)).toBe("00:00.0");
  });
});

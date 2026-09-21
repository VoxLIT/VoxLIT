import { describe, it, expect } from "vitest";
import { colorForClusterIndex, buildClusterColorMap } from "../clusterColors";

describe("clusterColors", () => {
  describe("colorForClusterIndex", () => {
    it("returns base palette color for indices within palette length (0 to 11)", () => {
      expect(colorForClusterIndex(0)).toBe("#2563eb");
      expect(colorForClusterIndex(1)).toBe("#dc2626");
      expect(colorForClusterIndex(11)).toBe("#be123c");
    });

    it("returns deterministic HSL color using golden-angle for indices >= 12", () => {
      const color12 = colorForClusterIndex(12);
      expect(color12).toMatch(/^hsl\(\d+(\.\d+)?, 65%, 45%\)$/);

      // Verify determinism
      expect(colorForClusterIndex(12)).toBe(color12);

      // Verify different indices generate different hues
      const color13 = colorForClusterIndex(13);
      expect(color13).not.toBe(color12);
    });
  });

  describe("buildClusterColorMap", () => {
    it("handles empty cluster labels list", () => {
      const map = buildClusterColorMap([]);
      expect(map).toEqual({});
    });

    it("assigns colors deterministically in order of first appearance", () => {
      const labels = ["Cluster 1", "Cluster 2", "Cluster 1", "Cluster 3"];
      const map = buildClusterColorMap(labels);

      expect(Object.keys(map)).toEqual(["Cluster 1", "Cluster 2", "Cluster 3"]);
      expect(map["Cluster 1"]).toBe(colorForClusterIndex(0));
      expect(map["Cluster 2"]).toBe(colorForClusterIndex(1));
      expect(map["Cluster 3"]).toBe(colorForClusterIndex(2));
    });

    it("handles more than 12 unique clusters gracefully using the golden-angle fallback", () => {
      const labels = Array.from({ length: 15 }, (_, i) => `Cluster ${i + 1}`);
      const map = buildClusterColorMap(labels);

      expect(Object.keys(map)).toHaveLength(15);
      expect(map["Cluster 13"]).toBe(colorForClusterIndex(12));
      expect(map["Cluster 13"]).toMatch(/^hsl\(/);
    });
  });
});

/**
 * The embedding view's colour ramp.
 *
 * This is the mechanism that lets the view show structure WITHOUT revealing
 * the dataset's labels: points are coloured by the detector's own score. If
 * the ramp ever stopped being a function of the score alone, that guarantee
 * would go with it.
 */
import { describe, expect, it } from "vitest";
import { BONAFIDE_COLOR, SPOOF_COLOR, spoofScoreColor } from "../embeddingColors";

describe("spoofScoreColor", () => {
  it("anchors the ends of the ramp on the two named colours", () => {
    expect(spoofScoreColor(0)).toBe(BONAFIDE_COLOR);
    expect(spoofScoreColor(1)).toBe(SPOOF_COLOR);
  });

  it("interpolates the midpoint between them", () => {
    // #2563eb -> #dc2626. Halfway, rounding each channel:
    //   R (0x25 + 0xdc) / 2 = 128.5 -> 0x81
    //   G (0x63 + 0x26) / 2 =  68.5 -> 0x45
    //   B (0xeb + 0x26) / 2 = 136.5 -> 0x89
    expect(spoofScoreColor(0.5)).toBe("#814589");
  });

  it("moves monotonically from blue to red", () => {
    const red = (hex: string) => parseInt(hex.slice(1, 3), 16);
    const scores = [0, 0.25, 0.5, 0.75, 1];
    const reds = scores.map((score) => red(spoofScoreColor(score)));
    expect(reds).toEqual([...reds].sort((a, b) => a - b));
  });

  it("clamps out-of-range and non-finite scores rather than emitting nonsense", () => {
    expect(spoofScoreColor(-2)).toBe(BONAFIDE_COLOR);
    expect(spoofScoreColor(7)).toBe(SPOOF_COLOR);
    expect(spoofScoreColor(Number.NaN)).toBe(BONAFIDE_COLOR);
  });

  it("always returns a six-digit hex, which is what the plot parses", () => {
    for (const score of [0, 0.01, 0.137, 0.5, 0.999, 1]) {
      expect(spoofScoreColor(score)).toMatch(/^#[0-9a-f]{6}$/);
    }
  });
});

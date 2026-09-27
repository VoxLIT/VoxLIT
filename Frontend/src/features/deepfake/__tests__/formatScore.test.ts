import { describe, expect, it } from "vitest";
import { formatScore } from "../page/palette";

describe("formatScore", () => {
  it("keeps the requested precision for ordinary scores", () => {
    expect(formatScore(0.072)).toBe("0.072");
    expect(formatScore(0.5)).toBe("0.500");
    expect(formatScore(0.28, 2)).toBe("0.28");
  });

  it("adds decimals until a saturated score is no longer 0 or 1", () => {
    expect(formatScore(0.000027)).toBe("0.000027");
    expect(formatScore(0.000406)).toBe("0.00041");
    expect(formatScore(0.999997)).toBe("0.999997");
    expect(formatScore(0.003386, 2)).toBe("0.0034");
  });

  it("bounds values past the backend's 6-decimal rounding", () => {
    expect(formatScore(0)).toBe("<0.000001");
    expect(formatScore(1)).toBe(">0.999999");
  });
});

import { describe, expect, it } from "vitest";
import { ASVSPOOF2019_LA_ATTACKS, attackShortLabel, describeAttack } from "../page/attacks";

describe("ASVspoof 2019 LA attack labels", () => {
  it("covers every evaluation attack, A07 to A19", () => {
    const ids = Array.from({ length: 13 }, (_, i) => `A${String(i + 7).padStart(2, "0")}`);
    expect(Object.keys(ASVSPOOF2019_LA_ATTACKS).sort()).toEqual(ids);
  });

  it("names the generator behind an id", () => {
    expect(attackShortLabel("A10")).toBe("TTS · WaveRNN");
    expect(describeAttack("A18")?.kind).toBe("VC");
  });

  it("accepts unpadded ids and leaves unknown ones alone", () => {
    expect(describeAttack("a7")?.vocoder).toBe("WORLD");
    expect(describeAttack("my-custom-attack")).toBeNull();
    expect(attackShortLabel("bonafide")).toBeNull();
  });
});

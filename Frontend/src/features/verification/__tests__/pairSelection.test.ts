import { describe, it, expect } from "vitest";
import { getNextVerificationPairSelection } from "../pairSelection";

describe("getNextVerificationPairSelection", () => {
  it("selects first clip when starting with no selection", () => {
    const result = getNextVerificationPairSelection([], "clip_a");
    expect(result).toEqual(["clip_a"]);
  });

  it("selects first clip when prevSelection is undefined", () => {
    const result = getNextVerificationPairSelection(undefined, "clip_a");
    expect(result).toEqual(["clip_a"]);
  });

  it("forms a pair when clicking a second clip", () => {
    const result = getNextVerificationPairSelection(["clip_a"], "clip_b");
    expect(result).toEqual(["clip_a", "clip_b"]);
  });

  it("removes clip A's circle and retains only clip B when clicking clip B again in pair [clip_a, clip_b]", () => {
    // User scenario: Click A -> Click B (pair active) -> Click B again
    // Expected: Pair comparison disables, clip A's selection is removed, only clip B remains.
    const result = getNextVerificationPairSelection(["clip_a", "clip_b"], "clip_b");
    expect(result).toEqual(["clip_b"]);
  });

  it("removes clip B's circle and retains only clip A when clicking clip A in pair [clip_a, clip_b]", () => {
    // User scenario: Click A -> Click B (pair active) -> Click A
    // Expected: Pair comparison disables, clip B's selection is removed, only clip A remains.
    const result = getNextVerificationPairSelection(["clip_a", "clip_b"], "clip_a");
    expect(result).toEqual(["clip_a"]);
  });

  it("slides window to [clip_b, clip_c] when clicking a third clip in pair [clip_a, clip_b]", () => {
    const result = getNextVerificationPairSelection(["clip_a", "clip_b"], "clip_c");
    expect(result).toEqual(["clip_b", "clip_c"]);
  });

  it("starts a fresh single-clip selection when clicking while 3+ clips are selected", () => {
    // A box/lasso group selection is replaced by the clicked clip, whether or
    // not that clip was part of the group.
    const group = ["clip_a", "clip_b", "clip_c", "clip_d"];
    expect(getNextVerificationPairSelection(group, "clip_e")).toEqual(["clip_e"]);
    expect(getNextVerificationPairSelection(group, "clip_c")).toEqual(["clip_c"]);
    expect(getNextVerificationPairSelection(["clip_a", "clip_b", "clip_c"], "clip_d")).toEqual(["clip_d"]);
  });

  it("retains single clip when clicking the same single clip again", () => {
    const result = getNextVerificationPairSelection(["clip_b"], "clip_b");
    expect(result).toEqual(["clip_b"]);
  });
});

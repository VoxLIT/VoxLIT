import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import type { DiarizationResult } from "../types";
import { SimilarityView } from "../page/SimilarityView";
import { result } from "./fixtures";

// Must match SimilarityView: default canvas size and axis strip width.
const SIZE = 480;
const STRIP = 8;

/** Client coordinates of the centre of cell (row, col) for an n×n matrix. */
const cellPoint = (row: number, col: number, n: number) => {
  const cell = (SIZE - STRIP) / n;
  return { clientX: STRIP + (col + 0.5) * cell, clientY: STRIP + (row + 0.5) * cell };
};

const setup = (run: DiarizationResult = result) => {
  const onPlayPair = vi.fn();
  const onPlay = vi.fn();
  const onSelect = vi.fn();
  render(<SimilarityView result={run} selectedId={null} onSelect={onSelect} onPlay={onPlay} onPlayPair={onPlayPair} />);
  return { canvas: getCanvas(), onPlayPair, onPlay, onSelect };
};

/** The canvas remounts when the order changes (to replay the wipe), so query
 *  it again after toggling. */
const getCanvas = () => {
  const canvas = screen.getByRole("img");
  vi.spyOn(canvas, "getBoundingClientRect").mockReturnValue({
    left: 0,
    top: 0,
    right: SIZE,
    bottom: SIZE,
    width: SIZE,
    height: SIZE,
    x: 0,
    y: 0,
    toJSON: () => ({}),
  });
  return canvas;
};

beforeEach(() => {
  // jsdom has no 2D context; the view skips drawing without one.
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue(null);
});
afterEach(() => vi.restoreAllMocks());

describe("SimilarityView", () => {
  it("plays the pair under a clicked cell, in time order", () => {
    const { canvas, onPlayPair } = setup();
    // 4 embedded segments (seg_5 is too short): row 0 = seg_1, col 2 = seg_3
    fireEvent.click(canvas, cellPoint(0, 2, 4));
    expect(onPlayPair).toHaveBeenCalledWith("seg_1", "seg_3");
  });

  it("keeps click-to-play correct after grouping by speaker", () => {
    const interleaved: DiarizationResult = {
      ...result,
      segments: result.segments.map((s) => (s.id === "seg_2" ? { ...s, speaker: "SPEAKER_01" } : s.id === "seg_3" ? { ...s, speaker: "SPEAKER_00" } : s)),
    };
    const { onPlayPair } = setup(interleaved);
    fireEvent.click(screen.getByRole("button", { name: /technical details/i }));
    fireEvent.click(screen.getByRole("radio", { name: "Speaker" }));
    // grouped: SPEAKER_00 → seg_1, seg_3; SPEAKER_01 → seg_2, seg_4
    fireEvent.click(getCanvas(), cellPoint(1, 2, 4));
    expect(onPlayPair).toHaveBeenCalledWith("seg_3", "seg_2");
  });

  it("explains a hovered pair in words and plays both from the popup", () => {
    const { canvas, onPlayPair } = setup();
    fireEvent.mouseMove(canvas, cellPoint(0, 3, 4)); // seg_1 vs seg_4, similarity ≈ 1
    const popup = screen.getByRole("dialog", { name: "Compare two segments" });
    expect(popup).toHaveTextContent("These two moments sound very alike.");
    fireEvent.click(screen.getByRole("button", { name: /play both/i }));
    expect(onPlayPair).toHaveBeenCalledWith("seg_1", "seg_4");
  });

  it("names the pair most likely to be mixed up", () => {
    setup();
    expect(
      screen.getByText("SPEAKER_00 and SPEAKER_01 sound the most alike, so these are the voices the system is most likely to mix up."),
    ).toBeInTheDocument();
  });

  it("lists uncertain segments with play buttons, and not the unscored one", () => {
    const { onPlay } = setup();
    const rows = screen.getAllByTestId("uncertain-row");
    expect(rows).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Play SPEAKER_01 at 00:11.0" }));
    expect(onPlay).toHaveBeenCalledWith("seg_4");
    expect(screen.getByText(/1 segment was too short to score/)).toBeInTheDocument();
  });
});

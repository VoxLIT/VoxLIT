/**
 * SegmentMap — Step 3 of the diarization page: selected segment | 2D map |
 * nearest segments. Covers the disabled 3D pill, the rule that selection is
 * shown by size and never by a new colour, neighbours coming from embeddings
 * rather than the map, and the side columns surviving a failed projection.
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { SegmentMap } from "../page/SegmentMap";
import { DARK_PALETTE } from "../page/palette";
import { projection, result } from "./fixtures";

const renderMap = (overrides: Partial<Parameters<typeof SegmentMap>[0]> = {}) => {
  const props = {
    result,
    projection,
    projectionError: null,
    isRunning: false,
    embeddingDimension: 256,
    selectedId: null,
    hoveredId: null,
    onHover: vi.fn(),
    onSelect: vi.fn(),
    onPlay: vi.fn(),
    ...overrides,
  };
  render(<SegmentMap {...props} />);
  return props;
};

const pointFor = (id: string) =>
  screen.getAllByTestId("map-point").find((node) => node.getAttribute("data-segment-id") === id)!;

describe("SegmentMap", () => {
  it("offers 2D only, with 3D shown disabled as coming soon", () => {
    renderMap();
    const group = screen.getByRole("radiogroup", { name: "Dimensions" });
    expect(within(group).getByRole("radio", { name: /2D/ })).toHaveAttribute("aria-checked", "true");
    const threeD = within(group).getByRole("radio", { name: /3D/ });
    expect(threeD).toHaveAttribute("aria-disabled", "true");
    expect(threeD).toHaveAttribute("aria-checked", "false");
    expect(threeD).toHaveAttribute("title", "coming soon");
  });

  it("draws one dot per embedded segment, in its speaker's colour", () => {
    renderMap();
    expect(screen.getAllByTestId("map-point")).toHaveLength(projection.points.length);
    const dot = pointFor("seg_3").querySelector('[data-testid="map-dot"]')!;
    expect(dot).toHaveAttribute("fill", DARK_PALETTE.speakerColor(result.speakers, "SPEAKER_01"));
  });

  it("marks the selected dot by size and ripple, keeping its speaker colour", () => {
    renderMap({ selectedId: "seg_1" });
    const point = pointFor("seg_1");
    expect(point).toHaveAttribute("data-selected", "true");
    expect(point.querySelector(".dz-ripple")).not.toBeNull();
    expect(point.querySelector('[data-testid="map-dot"]')).toHaveAttribute(
      "fill",
      DARK_PALETTE.speakerColor(result.speakers, "SPEAKER_00"),
    );
  });

  it("lists nearest segments by embedding similarity and flags other speakers", () => {
    renderMap({ selectedId: "seg_1" });
    const rows = screen.getAllByTestId("neighbour-row");
    expect(rows).toHaveLength(3);
    // seg_4 is SPEAKER_01 and closest in embedding space, although far on the map.
    expect(within(rows[0]).getByText("SPEAKER_01")).toBeInTheDocument();
    expect(within(rows[0]).getByText(/different speaker/)).toBeInTheDocument();
    expect(within(rows[1]).getByText(/same speaker/)).toBeInTheDocument();
    expect(screen.getByText("Its closest match belongs to SPEAKER_01.")).toBeInTheDocument();
    expect(screen.getAllByTestId("neighbour-line")).toHaveLength(3);
  });

  it("plays a neighbour without changing the selection", () => {
    const props = renderMap({ selectedId: "seg_1" });
    const rows = screen.getAllByTestId("neighbour-row");
    fireEvent.click(within(rows[0]).getByRole("button", { name: /^Play/ }));
    expect(props.onPlay).toHaveBeenCalledWith("seg_4");
    expect(props.onSelect).not.toHaveBeenCalled();
  });

  it("explains a segment too short to embed instead of inventing neighbours", () => {
    renderMap({ selectedId: "seg_5" });
    expect(screen.getByText(/too short to have a voice fingerprint/i)).toBeInTheDocument();
    expect(screen.queryAllByTestId("neighbour-row")).toHaveLength(0);
  });

  it("keeps both side columns working when the projection failed", () => {
    renderMap({
      projection: null,
      projectionError: "Not enough embeddable segments for a 2D projection.",
      selectedId: "seg_1",
    });
    expect(screen.queryAllByTestId("map-point")).toHaveLength(0);
    expect(screen.getByText(/couldn.t lay out the map/i)).toHaveTextContent("Not enough embeddable segments");
    expect(screen.getByRole("complementary", { name: "Selected segment" })).toHaveTextContent("SPEAKER_00");
    expect(screen.getAllByTestId("neighbour-row")).toHaveLength(3);
  });

  it("opens a hover card and shares the hover with the timeline", async () => {
    const props = renderMap();
    fireEvent.mouseEnter(pointFor("seg_2").querySelector('circle[role="button"]')!);
    expect(props.onHover).toHaveBeenCalledWith("seg_2");
    const card = await screen.findByRole("dialog", { name: /SPEAKER_00, 00:03.5/ });
    expect(within(card).getByText("The model was fairly sure.")).toBeInTheDocument();
  });
});

/**
 * SegmentMap — Step 3 of the diarization page: selected segment | 2D/3D map |
 * nearest segments. Covers the 2D/3D switch, the rule that selection is
 * shown by size and never by a new colour, neighbours coming from embeddings
 * rather than the map, and the side columns surviving a failed projection.
 */
import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import { SegmentMap } from "../page/SegmentMap";
import { DARK_PALETTE } from "../page/palette";
import { projection, projection3d, result } from "./fixtures";

// jsdom has no WebGL, so the three.js scene is replaced by a stand-in that
// exposes what SegmentMap hands it.
vi.mock("../page/SegmentMap3D", () => ({
  default: ({
    points,
    neighbourIds,
    onSelect,
    onHover,
  }: {
    points: { id: string; z?: number }[];
    neighbourIds: string[];
    onSelect: (id: string) => void;
    onHover: (event: { segmentId: string; clientX: number; clientY: number } | null) => void;
  }) => (
    <div data-testid="segment-map-3d" data-neighbours={neighbourIds.join(",")}>
      {points.map((point) => (
        <button
          key={point.id}
          type="button"
          data-testid="map3d-point"
          data-z={point.z}
          onClick={() => onSelect(point.id)}
          onMouseEnter={() => onHover({ segmentId: point.id, clientX: 10, clientY: 10 })}
        >
          {point.id}
        </button>
      ))}
    </div>
  ),
}));

const renderMap = (overrides: Partial<Parameters<typeof SegmentMap>[0]> = {}) => {
  const props = {
    result,
    projection,
    projectionError: null,
    dims: 2 as const,
    onDimsChange: vi.fn(),
    projection3d: null,
    projection3dError: null,
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
  it("starts in 2D and asks for 3D when the 3D pill is clicked", () => {
    const props = renderMap();
    const group = screen.getByRole("radiogroup", { name: "Dimensions" });
    expect(within(group).getByRole("radio", { name: /2D/ })).toHaveAttribute("aria-checked", "true");
    const threeD = within(group).getByRole("radio", { name: /3D/ });
    expect(threeD).toHaveAttribute("aria-checked", "false");
    expect(threeD).not.toHaveAttribute("aria-disabled");
    fireEvent.click(threeD);
    expect(props.onDimsChange).toHaveBeenCalledWith(3);
  });

  it("says the 3D layout is on its way until it arrives", () => {
    renderMap({ dims: 3 });
    expect(screen.getByRole("status")).toHaveTextContent(/laying out the 3d map/i);
    expect(screen.queryAllByTestId("map-point")).toHaveLength(0);
  });

  it("draws the 3D map from the 3D projection, with embedding-space neighbours", async () => {
    const props = renderMap({ dims: 3, projection3d, selectedId: "seg_1" });
    const scene = await screen.findByTestId("segment-map-3d");
    const balls = within(scene).getAllByTestId("map3d-point");
    expect(balls).toHaveLength(projection3d.points.length);
    expect(balls[0]).toHaveAttribute("data-z", "-1.5");
    // Same neighbours as the side list — cosine in the model's space, not map distance.
    expect(scene).toHaveAttribute("data-neighbours", "seg_4,seg_2,seg_3");
    expect(screen.queryAllByTestId("map-point")).toHaveLength(0);
    fireEvent.click(balls[2]);
    expect(props.onSelect).toHaveBeenCalledWith("seg_3");
  });

  it("opens the same hover card from a 3D ball", async () => {
    const props = renderMap({ dims: 3, projection3d });
    fireEvent.mouseEnter(await screen.findByRole("button", { name: "seg_2" }));
    expect(props.onHover).toHaveBeenCalledWith("seg_2");
    expect(await screen.findByRole("dialog", { name: /SPEAKER_00, 00:03.5/ })).toBeInTheDocument();
  });

  it("shows a failed 3D layout without touching the side columns", () => {
    renderMap({
      dims: 3,
      projection3dError: "Not enough embeddable segments for a 3D projection.",
      selectedId: "seg_1",
    });
    expect(screen.getByText(/couldn.t lay out the map/i)).toHaveTextContent("3D projection");
    expect(screen.getAllByTestId("neighbour-row")).toHaveLength(3);
  });

  it("reports how much of the variation the 3D axes keep", () => {
    renderMap({ dims: 3, projection3d });
    fireEvent.click(screen.getByRole("button", { name: /technical details/i }));
    expect(screen.getByText(/projected to 3D with PCA/)).toBeInTheDocument();
    expect(screen.getByText(/These 3 axes keep 92% of the variation/)).toHaveTextContent(
      "axis 1: 50%, axis 2: 30%, axis 3: 12%",
    );
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

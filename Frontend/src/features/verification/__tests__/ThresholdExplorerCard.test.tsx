import { afterEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";
import React from "react";
import type { BatchAnalysisResponse } from "../batchTypes";

// Plotly needs a real layout engine; what matters here is the data handed to it.
const plotProps: { data: Record<string, any>[]; layout: Record<string, any> }[] = [];
vi.mock("react-plotly.js", () => ({
  default: (props: { data: Record<string, any>[]; layout: Record<string, any> }) => {
    plotProps.push(props);
    return <div data-testid="plot" />;
  },
}));

const { ThresholdExplorerCard } = await import("../ThresholdExplorerCard");
const { PairComparisonCard } = await import("../PairComparisonCard");

const CALIBRATED = 0.5;
const labels = ["rec_a", "rec_b", "rec_c", "rec_d"];

const clusterStats = (cluster_id: string) => ({
  cluster_id,
  mean_similarity_to_cluster: null,
  min_similarity_to_cluster: null,
  nearest_index: 0,
  nearest_label: "rec_a",
  nearest_similarity: 0.5,
  nearest_in_same_cluster: true,
});

// Recordings a-c are speaker A, d is speaker B.
// Same-speaker similarities: 0.9, 0.8, 0.4. Different-speaker: 0.6, 0.2, 0.1.
const similarity_matrix = [
  [1.0, 0.9, 0.8, 0.6],
  [0.9, 1.0, 0.4, 0.2],
  [0.8, 0.4, 1.0, 0.1],
  [0.6, 0.2, 0.1, 1.0],
];

const batchResult: BatchAnalysisResponse = {
  model: "ecapa-tdnn",
  model_label: "ECAPA-TDNN",
  threshold: CALIBRATED,
  recording_count: 4,
  embedding_dimension: 2,
  labels,
  ground_truth_groups: ["A", "A", "A", "B"],
  ground_truth_available: true,
  embeddings: [],
  similarity_matrix,
  decision_matrix: similarity_matrix.map((row) => row.map((value) => value >= CALIBRATED)),
  clustering_distance_threshold: 0.65,
  clustering_threshold_version: "1.0",
  cluster_labels: ["cluster-1", "cluster-1", "cluster-1", "cluster-2"],
  cluster_fit_scores: [1, 1, 1, 1],
  cluster_count: 2,
  cluster_summaries: [],
  recording_cluster_stats: [
    clusterStats("cluster-1"),
    clusterStats("cluster-1"),
    clusterStats("cluster-1"),
    clusterStats("cluster-2"),
  ],
  evaluation_metrics: null,
  true_speaker_count: 2,
};

const noGroundTruth: BatchAnalysisResponse = {
  ...batchResult,
  ground_truth_groups: null,
  ground_truth_available: false,
  true_speaker_count: null,
};

const displayNames: Record<string, string> = {
  rec_a: "a.wav",
  rec_b: "b.wav",
  rec_c: "c.wav",
  rec_d: "d.wav",
};
const resolveLabel = (id: string) => displayNames[id] ?? id;

const slider = () => screen.getByRole("slider", { name: "What-if threshold" }) as HTMLInputElement;
const moveSlider = (value: number) => fireEvent.change(slider(), { target: { value: String(value) } });

const lastPlotNamed = (traceName: string) => {
  for (let index = plotProps.length - 1; index >= 0; index -= 1) {
    if (plotProps[index].data.some((trace) => trace.name === traceName)) return plotProps[index];
  }
  throw new Error(`No plot with a trace named ${traceName}`);
};

const deepFreeze = <T,>(value: T): T => {
  if (value && typeof value === "object") {
    Object.values(value as object).forEach(deepFreeze);
    Object.freeze(value);
  }
  return value;
};

describe("ThresholdExplorerCard", () => {
  afterEach(() => {
    plotProps.length = 0;
  });

  it("renders the slider, charts and live stats with ground truth", () => {
    render(<ThresholdExplorerCard batchResult={batchResult} resolveLabel={resolveLabel} />);

    expect(screen.getByText("Threshold explorer")).toBeInTheDocument();
    expect(slider().value).toBe("0.5");
    expect(slider().min).toBe("0.1");
    expect(slider().max).toBe("0.9");
    expect(slider().step).toBe("0.01");
    expect(screen.getByTestId("whatif-threshold")).toHaveTextContent("0.5000");
    expect(screen.getByTestId("calibrated-threshold")).toHaveTextContent("0.5000");
    expect(screen.getAllByTestId("plot")).toHaveLength(2);

    const stats = screen.getByTestId("threshold-stats");
    expect(stats).toHaveTextContent("Accepted pairs: 3 of 6");
    expect(stats).toHaveTextContent("False accepts: 1 (FAR 33.3%)");
    expect(stats).toHaveTextContent("False rejects: 1 (FRR 33.3%)");
    expect(stats).toHaveTextContent("Accuracy: 66.7%");
    expect(stats).toHaveTextContent("Balanced accuracy: 66.7%");
    // Balanced accuracy is the primary stat, so it comes first.
    expect(stats.textContent?.startsWith("Balanced accuracy")).toBe(true);
    expect(stats).toHaveTextContent("EER 33.3% at threshold 0.6000");
    expect(screen.getByTestId("class-counts")).toHaveTextContent("3 same-speaker pairs · 3 different-speaker pairs");

    expect(screen.getByTestId("changed-pairs")).toHaveTextContent("No decisions change");
    expect(screen.queryByText(/Error rates need known speaker groups/)).not.toBeInTheDocument();
  });

  it("draws green/red histograms, a dashed calibrated line and a solid what-if line", () => {
    render(<ThresholdExplorerCard batchResult={batchResult} resolveLabel={resolveLabel} />);
    moveSlider(0.3);

    const histogram = lastPlotNamed("Same speaker");
    const same = histogram.data.find((trace) => trace.name === "Same speaker");
    const different = histogram.data.find((trace) => trace.name === "Different speaker");
    expect(same?.type).toBe("histogram");
    expect(same?.x).toEqual([0.9, 0.8, 0.4]);
    expect(same?.marker.color).toBe("#10b981");
    expect(different?.x).toEqual([0.6, 0.2, 0.1]);
    expect(different?.marker.color).toBe("#f43f5e");
    expect(histogram.layout.barmode).toBe("overlay");
    expect(histogram.layout.shapes).toHaveLength(2);
    expect(histogram.layout.shapes[0]).toMatchObject({ x0: CALIBRATED, x1: CALIBRATED, line: { dash: "dash" } });
    expect(histogram.layout.shapes[1]).toMatchObject({ x0: 0.3, x1: 0.3, line: { dash: "solid" } });

    const errorRates = lastPlotNamed("FAR");
    expect(errorRates.data.map((trace) => trace.name)).toEqual(["FAR", "FRR", "EER"]);
    const eer = errorRates.data.find((trace) => trace.name === "EER");
    expect(eer?.x).toEqual([0.6]);
    expect(eer?.y[0]).toBeCloseTo(1 / 3);
    expect(errorRates.layout.shapes).toHaveLength(1);
    expect(errorRates.layout.shapes[0]).toMatchObject({ x0: 0.3, x1: 0.3, line: { dash: "solid" } });
  });

  it("updates the stats and the changed-pairs list as the slider moves", () => {
    render(<ThresholdExplorerCard batchResult={batchResult} resolveLabel={resolveLabel} />);

    moveSlider(0.3);
    expect(screen.getByTestId("whatif-threshold")).toHaveTextContent("0.3000");
    expect(screen.getByTestId("calibrated-threshold")).toHaveTextContent("0.5000");
    expect(screen.getByTestId("threshold-stats")).toHaveTextContent("Accepted pairs: 4 of 6");
    expect(screen.getByTestId("threshold-stats")).toHaveTextContent("False rejects: 0 (FRR 0.0%)");

    let changed = screen.getByTestId("changed-pairs");
    expect(within(changed).getAllByRole("listitem")).toHaveLength(1);
    expect(changed).toHaveTextContent("b.wav ↔ c.wav");
    expect(changed).toHaveTextContent("0.4000");
    expect(changed).toHaveTextContent("Different speakers → Same speaker");

    moveSlider(0.85);
    changed = screen.getByTestId("changed-pairs");
    const rows = within(changed).getAllByRole("listitem");
    expect(rows).toHaveLength(2);
    expect(rows[0]).toHaveTextContent("a.wav ↔ d.wav");
    expect(rows[0]).toHaveTextContent("Same speaker → Different speakers");
    expect(rows[1]).toHaveTextContent("a.wav ↔ c.wav");
    expect(changed).not.toHaveTextContent("b.wav ↔ c.wav");
  });

  it("caps the list at ten rows and says how many changed", () => {
    // Six recordings of one speaker -> 15 pairs: one at 0.9, fourteen at 0.4.
    const size = 6;
    const many: BatchAnalysisResponse = {
      ...batchResult,
      threshold: 0.3,
      labels: Array.from({ length: size }, (_, index) => `rec_${index}`),
      ground_truth_groups: Array.from({ length: size }, () => "A"),
      similarity_matrix: Array.from({ length: size }, (_, i) =>
        Array.from({ length: size }, (_, j) => (i === j ? 1 : i + j === 1 ? 0.9 : 0.4))
      ),
    };
    render(<ThresholdExplorerCard batchResult={many} />);

    moveSlider(0.5);

    const changed = screen.getByTestId("changed-pairs");
    expect(within(changed).getAllByRole("listitem")).toHaveLength(10);
    expect(changed).toHaveTextContent("Showing 10 of 14 changed pairs.");
  });

  it("shares one fixed x range and the same plot margins between the slider and both charts", () => {
    render(<ThresholdExplorerCard batchResult={batchResult} resolveLabel={resolveLabel} />);
    moveSlider(0.3);

    const histogram = lastPlotNamed("Same speaker").layout;
    const errorRates = lastPlotNamed("FAR").layout;
    for (const layout of [histogram, errorRates]) {
      expect(layout.xaxis.range).toEqual([Number(slider().min), Number(slider().max)]);
      expect(layout.xaxis.autorange).toBe(false);
      expect(layout.autosize).toBe(true);
      expect(layout.margin).toMatchObject({ l: 56, r: 16 });
    }

    // The slider row holds only the slider, padded to the plot margins.
    const sliderRow = screen.getByTestId("threshold-slider-row");
    expect(sliderRow).toHaveStyle({ paddingLeft: "56px", paddingRight: "16px" });
    expect(sliderRow.children).toHaveLength(1);
    expect(sliderRow.firstElementChild).toBe(slider());
    expect(within(sliderRow).queryByRole("button")).not.toBeInTheDocument();
  });

  it("Reset to calibrated sits in the top row and restores the calibrated value", () => {
    render(<ThresholdExplorerCard batchResult={batchResult} resolveLabel={resolveLabel} />);
    const headerRow = screen.getByTestId("threshold-header-row");
    const reset = within(headerRow).getByRole("button", { name: "Reset to calibrated" });
    expect(headerRow).toContainElement(screen.getByTestId("calibrated-threshold"));
    expect(reset).toBeDisabled();

    moveSlider(0.3);
    expect(reset).not.toBeDisabled();
    expect(within(screen.getByTestId("changed-pairs")).getAllByRole("listitem")).toHaveLength(1);

    fireEvent.click(reset);

    expect(slider().value).toBe("0.5");
    expect(screen.getByTestId("whatif-threshold")).toHaveTextContent("0.5000");
    expect(screen.getByTestId("threshold-stats")).toHaveTextContent("Accepted pairs: 3 of 6");
    expect(screen.getByTestId("changed-pairs")).toHaveTextContent("No decisions change");
    expect(reset).toBeDisabled();
  });

  it("shows only the all-pairs histogram, slider and accepted count without ground truth", () => {
    render(<ThresholdExplorerCard batchResult={noGroundTruth} resolveLabel={resolveLabel} />);

    expect(
      screen.getByText("Error rates need known speaker groups, which this dataset does not have.")
    ).toBeInTheDocument();
    expect(screen.getAllByTestId("plot")).toHaveLength(1);
    expect(plotProps[plotProps.length - 1].data.map((trace) => trace.name)).toEqual(["All pairs"]);
    expect(plotProps[plotProps.length - 1].data[0].x).toHaveLength(6);

    const stats = screen.getByTestId("threshold-stats");
    expect(stats).toHaveTextContent("Accepted pairs: 3 of 6");
    expect(stats).not.toHaveTextContent("FAR");
    expect(stats).not.toHaveTextContent("EER");
    expect(screen.queryByTestId("changed-pairs")).not.toBeInTheDocument();

    moveSlider(0.3);
    expect(screen.getByTestId("threshold-stats")).toHaveTextContent("Accepted pairs: 4 of 6");
  });

  it("shows n/a rather than NaN when a ground-truth class is empty", () => {
    const oneSpeaker: BatchAnalysisResponse = { ...batchResult, ground_truth_groups: ["A", "A", "A", "A"] };
    render(<ThresholdExplorerCard batchResult={oneSpeaker} resolveLabel={resolveLabel} />);

    const stats = screen.getByTestId("threshold-stats");
    expect(stats).toHaveTextContent("FAR n/a");
    expect(stats).toHaveTextContent("Balanced accuracy: n/a");
    expect(screen.getByTestId("class-counts")).toHaveTextContent("6 same-speaker pairs · 0 different-speaker pairs");
    expect(stats).toHaveTextContent("EER: n/a");
    expect(stats).not.toHaveTextContent("NaN");
  });

  it("starts at the calibrated threshold even when it lies outside the batch's similarity range", () => {
    render(<ThresholdExplorerCard batchResult={{ ...batchResult, threshold: 0.95 }} resolveLabel={resolveLabel} />);

    expect(slider().value).toBe("0.95");
    expect(slider().max).toBe("0.95");
    expect(screen.getByTestId("threshold-stats")).toHaveTextContent("Accepted pairs: 0 of 6");
  });

  it("leaves the batch result and the other cards unaffected", () => {
    const frozen = deepFreeze(structuredClone(batchResult));
    const before = JSON.stringify(frozen);
    const labelToIndex = new Map(labels.map((label, index) => [label, index]));

    render(
      <>
        <div data-testid="pair-card">
          <PairComparisonCard
            selectedLabels={["rec_b", "rec_c"]}
            batchResult={frozen}
            labelToIndex={labelToIndex}
            resolveLabel={resolveLabel}
          />
        </div>
        <ThresholdExplorerCard batchResult={frozen} resolveLabel={resolveLabel} />
      </>
    );

    const pairCard = screen.getByTestId("pair-card");
    const pairCardBefore = pairCard.innerHTML;
    // b/c sits at 0.4: rejected at the calibrated 0.5.
    expect(pairCard).toHaveTextContent("Different speakers");
    expect(pairCard).toHaveTextContent("0.5000");

    // The what-if threshold now accepts that pair...
    moveSlider(0.3);
    expect(screen.getByTestId("changed-pairs")).toHaveTextContent("b.wav ↔ c.wav");

    // ...but the pair card still shows the calibrated threshold and decision.
    expect(pairCard.innerHTML).toBe(pairCardBefore);
    expect(JSON.stringify(frozen)).toBe(before);
    expect(frozen.threshold).toBe(CALIBRATED);
  });
});

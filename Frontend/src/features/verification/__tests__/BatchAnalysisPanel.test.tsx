import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import type { BatchAnalysisResponse } from "../batchTypes";

// The panel is exercised for its card wiring only, so its children and the
// shared context are stubbed.
vi.mock("@/tasks/registry", () => ({ VERIFICATION_DEMO_DATASET_ID: "voxceleb1-indian-demo" }));
vi.mock("@/contexts/EmbeddingContext", () => ({
  useEmbedding: () => ({
    setEmbeddingDataDirect: () => {},
    focusedClusterId: null,
    setFocusedClusterId: () => {},
  }),
}));
vi.mock("../PairComparisonCard", () => ({
  PairComparisonCard: ({ selectedLabels }: { selectedLabels: string[] }) => (
    <div data-testid="pair-card">{selectedLabels.join(",")}</div>
  ),
}));
vi.mock("../ClusterSummaryList", () => ({ ClusterSummaryList: () => <div data-testid="cluster-summary" /> }));
const explorerProps: { selectedIndices?: number[] }[] = [];
vi.mock("../ThresholdExplorerCard", () => ({
  ThresholdExplorerCard: (props: { selectedIndices?: number[] }) => {
    explorerProps.push(props);
    return <div data-testid="threshold-explorer" />;
  },
}));

const { BatchAnalysisPanel } = await import("../BatchAnalysisPanel");

const labels = ["rec_a", "rec_b", "rec_c", "rec_d"];
const batchResult = {
  model: "ecapa-tdnn",
  model_label: "ECAPA-TDNN",
  threshold: 0.5,
  labels,
  cluster_labels: ["cluster-1", "cluster-1", "cluster-1", "cluster-2"],
  cluster_summaries: [],
  recording_cluster_stats: [],
} as unknown as BatchAnalysisResponse;

const selectedBatchIds = [...labels];
const uploadedRawFiles = {};
const noop = () => {};

const renderPanel = (pairSelection: string[]) => (
  <BatchAnalysisPanel
    model="ecapa-tdnn"
    modelLabel="ECAPA-TDNN"
    dataset="my-dataset"
    originalDataset="my-dataset"
    uploadedRawFiles={uploadedRawFiles}
    selectedBatchIds={selectedBatchIds}
    pairSelection={pairSelection}
    selectedFile={null}
    onReprojectHandlerChange={noop}
    onLabelResolverChange={noop}
  />
);

const lastSelectedIndices = () => explorerProps[explorerProps.length - 1].selectedIndices;

describe("BatchAnalysisPanel selection routing", () => {
  beforeEach(() => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: true, status: 200, json: async () => batchResult }))
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    explorerProps.length = 0;
  });

  const renderWithResult = async (pairSelection: string[]) => {
    const view = render(renderPanel(pairSelection));
    fireEvent.click(screen.getByRole("button", { name: "Run batch analysis" }));
    await screen.findByTestId("threshold-explorer");
    return view;
  };

  it("shows the Pair Comparison card for exactly two selected points", async () => {
    await renderWithResult(["rec_b", "rec_d"]);

    expect(screen.getByTestId("pair-card")).toHaveTextContent("rec_b,rec_d");
    expect(lastSelectedIndices()).toBeUndefined();
  });

  it("keeps the Pair Comparison card and passes no indices for zero or one selected point", async () => {
    const { rerender } = await renderWithResult([]);
    expect(screen.getByTestId("pair-card")).toBeInTheDocument();
    expect(lastSelectedIndices()).toBeUndefined();

    rerender(renderPanel(["rec_c"]));
    expect(screen.getByTestId("pair-card")).toBeInTheDocument();
    expect(lastSelectedIndices()).toBeUndefined();
  });

  it("hides the Pair Comparison card and hands the batch indices to the explorer for 3+ points", async () => {
    const { rerender } = await renderWithResult(["rec_d", "rec_a", "rec_c"]);

    expect(screen.queryByTestId("pair-card")).not.toBeInTheDocument();
    expect(lastSelectedIndices()).toEqual([3, 0, 2]);

    // Labels that are not in the batch are ignored.
    rerender(renderPanel(["rec_a", "not_in_batch", "rec_b", "rec_c", "rec_d"]));
    expect(screen.queryByTestId("pair-card")).not.toBeInTheDocument();
    expect(lastSelectedIndices()).toEqual([0, 1, 2, 3]);

    // Back to a pair: the card returns and the explorer loses its selection.
    rerender(renderPanel(["rec_a", "rec_b"]));
    expect(screen.getByTestId("pair-card")).toHaveTextContent("rec_a,rec_b");
    expect(lastSelectedIndices()).toBeUndefined();
  });
});

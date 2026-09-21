import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import React from "react";
import { PairComparisonCard } from "../PairComparisonCard";
import type { BatchAnalysisResponse } from "../batchTypes";

const mockBatchResult: BatchAnalysisResponse = {
  model: "ecapa-tdnn",
  model_label: "ECAPA-TDNN",
  threshold: 0.35,
  clustering_distance_threshold: 0.65,
  clustering_threshold_version: "1.0",
  audio_count: 3,
  labels: ["Clip A", "Clip B", "Clip C"],
  recording_ids: ["rec_a", "rec_b", "rec_c"],
  similarity_matrix: [
    [1.0, 0.75, 0.2],
    [0.75, 1.0, 0.15],
    [0.2, 0.15, 1.0],
  ],
  decision_matrix: [
    [true, true, false],
    [true, true, false],
    [false, false, true],
  ],
  recording_cluster_stats: [
    {
      recording_id: "rec_a",
      label: "Clip A",
      cluster_id: "Cluster 1",
      silhouette_score: 0.8,
      mean_similarity_to_cluster: 0.75,
      min_similarity_to_cluster: 0.75,
      nearest_recording_id: "rec_b",
      nearest_label: "Clip B",
      nearest_similarity: 0.75,
      nearest_in_same_cluster: true,
      cluster_fit_margin: 0.4,
      intra_cluster_spread: 0.0,
    },
    {
      recording_id: "rec_b",
      label: "Clip B",
      cluster_id: "Cluster 1",
      silhouette_score: 0.8,
      mean_similarity_to_cluster: 0.75,
      min_similarity_to_cluster: 0.75,
      nearest_recording_id: "rec_a",
      nearest_label: "Clip A",
      nearest_similarity: 0.75,
      nearest_in_same_cluster: true,
      cluster_fit_margin: 0.4,
      intra_cluster_spread: 0.0,
    },
    {
      recording_id: "rec_c",
      label: "Clip C",
      cluster_id: "Cluster 2",
      silhouette_score: 0.9,
      mean_similarity_to_cluster: null,
      min_similarity_to_cluster: null,
      nearest_recording_id: "rec_a",
      nearest_label: "Clip A",
      nearest_similarity: 0.2,
      nearest_in_same_cluster: false,
      cluster_fit_margin: 0.8,
      intra_cluster_spread: null,
    },
  ],
  cluster_summaries: [],
  evaluation_metrics: null,
  ground_truth_available: false,
  anonymous_speaker_groups: null,
  per_clip_statistics: [],
  per_cluster_statistics: [],
  projections: {
    pca_2d: [[0, 0], [0, 0], [0, 0]],
    pca_3d: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    tsne_2d: [[0, 0], [0, 0], [0, 0]],
    tsne_3d: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
    umap_2d: [[0, 0], [0, 0], [0, 0]],
    umap_3d: [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
  },
};

const labelMap = new Map<string, number>([
  ["Clip A", 0],
  ["Clip B", 1],
  ["Clip C", 2],
]);

describe("PairComparisonCard", () => {
  it("renders selection prompt when fewer than 2 points are selected", () => {
    render(
      <PairComparisonCard
        selectedLabels={["Clip A"]}
        batchResult={mockBatchResult}
        labelToIndex={labelMap}
      />
    );
    expect(
      screen.getByText("Select exactly two points in the plot to compare.")
    ).toBeInTheDocument();
  });

  it("renders selection prompt when more than 2 points are selected", () => {
    render(
      <PairComparisonCard
        selectedLabels={["Clip A", "Clip B", "Clip C"]}
        batchResult={mockBatchResult}
        labelToIndex={labelMap}
      />
    );
    expect(
      screen.getByText("Select exactly two points in the plot to compare.")
    ).toBeInTheDocument();
  });

  it("renders invalid selection notice if points cannot be resolved in labelToIndex", () => {
    render(
      <PairComparisonCard
        selectedLabels={["Clip A", "NonExistent Clip"]}
        batchResult={mockBatchResult}
        labelToIndex={labelMap}
      />
    );
    expect(
      screen.getByText("Selection is no longer valid for the current batch.")
    ).toBeInTheDocument();
  });

  it("renders pair details, cosine similarity, threshold margin, and same speaker call for matching pair", () => {
    render(
      <PairComparisonCard
        selectedLabels={["Clip A", "Clip B"]}
        batchResult={mockBatchResult}
        labelToIndex={labelMap}
      />
    );

    // Similarity is 0.7500
    expect(screen.getByText("0.7500")).toBeInTheDocument();
    // Threshold is 0.3500
    expect(screen.getByText("0.3500")).toBeInTheDocument();
    // Margin is +0.4000
    expect(screen.getByText("+0.4000")).toBeInTheDocument();

    // Decisions
    expect(screen.getByText("Same speaker")).toBeInTheDocument();
    expect(screen.getByText("Same cluster")).toBeInTheDocument();
  });

  it("renders different speakers and different clusters for non-matching pair", () => {
    render(
      <PairComparisonCard
        selectedLabels={["Clip A", "Clip C"]}
        batchResult={mockBatchResult}
        labelToIndex={labelMap}
      />
    );

    // Similarity is 0.2000, margin is -0.1500
    expect(screen.getByText("0.2000")).toBeInTheDocument();
    expect(screen.getByText("-0.1500")).toBeInTheDocument();

    expect(screen.getByText("Different speakers")).toBeInTheDocument();
    expect(screen.getByText("Different clusters")).toBeInTheDocument();
  });

  it("renders warning banner when pair-verification decision and cluster assignment disagree", () => {
    // Construct case where similarity >= threshold (decision=true), but different clusters
    const disagreeBatch: BatchAnalysisResponse = {
      ...mockBatchResult,
      decision_matrix: [
        [true, true, true], // Clip A and C called same speaker
        [true, true, false],
        [true, false, true],
      ],
      // but Clip A is in Cluster 1 and Clip C is in Cluster 2
    };

    render(
      <PairComparisonCard
        selectedLabels={["Clip A", "Clip C"]}
        batchResult={disagreeBatch}
        labelToIndex={labelMap}
      />
    );

    expect(screen.getByText("Same speaker")).toBeInTheDocument();
    expect(screen.getByText("Different clusters")).toBeInTheDocument();
    expect(
      screen.getByText(/Pair verification and clustering use separately calibrated thresholds/i)
    ).toBeInTheDocument();
  });
});

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import React from "react";
import { ClusterSummaryList } from "../ClusterSummaryList";
import type { ClusterSummary } from "../batchTypes";

const mockSummaries: ClusterSummary[] = [
  {
    cluster_id: "Cluster 2",
    member_count: 2,
    representative_recording_id: "rec_2",
    representative_label: "Representative B",
    member_recording_ids: ["rec_2", "rec_3"],
    member_labels: ["Clip 2", "Clip 3"],
    mean_intra_cluster_similarity: 0.825,
    min_intra_cluster_similarity: 0.81,
    mean_fit_score: 0.85,
  },
  {
    cluster_id: "Cluster 1",
    member_count: 5,
    representative_recording_id: "rec_1",
    representative_label: "Representative A",
    member_recording_ids: ["rec_1", "rec_4", "rec_5", "rec_6", "rec_7"],
    member_labels: ["Clip 1", "Clip 4", "Clip 5", "Clip 6", "Clip 7"],
    mean_intra_cluster_similarity: 0.912,
    min_intra_cluster_similarity: 0.875,
    mean_fit_score: 0.89,
  },
  {
    cluster_id: "Cluster 3",
    member_count: 1,
    representative_recording_id: "rec_8",
    representative_label: "Representative C",
    member_recording_ids: ["rec_8"],
    member_labels: ["Clip 8"],
    mean_intra_cluster_similarity: null,
    min_intra_cluster_similarity: null,
    mean_fit_score: 1.0,
  },
];

const mockColorMap: Record<string, string> = {
  "Cluster 1": "#2563eb",
  "Cluster 2": "#dc2626",
  "Cluster 3": "#16a34a",
};

describe("ClusterSummaryList", () => {
  it("renders empty state when clusterSummaries is empty", () => {
    render(<ClusterSummaryList clusterSummaries={[]} clusterColorMap={{}} />);
    expect(screen.getByText("No speakers to display.")).toBeInTheDocument();
  });

  it("sorts clusters by member count descending", () => {
    render(
      <ClusterSummaryList
        clusterSummaries={mockSummaries}
        clusterColorMap={mockColorMap}
      />
    );

    const clusterButtons = screen.getAllByRole("button");
    expect(clusterButtons).toHaveLength(3);

    // Cluster 1 has 5 members, should be first
    expect(clusterButtons[0]).toHaveTextContent("Speaker 1");
    // Cluster 2 has 2 members, should be second
    expect(clusterButtons[1]).toHaveTextContent("Speaker 2");
    // Cluster 3 has 1 member, should be third
    expect(clusterButtons[2]).toHaveTextContent("Speaker 3");
  });

  it("renders 'Not applicable' for single-clip clusters with null similarities", () => {
    render(
      <ClusterSummaryList
        clusterSummaries={mockSummaries}
        clusterColorMap={mockColorMap}
      />
    );

    // For Cluster 3 (1 member), mean and min intra-cluster similarities are null
    expect(screen.getAllByText(/Not applicable/i).length).toBeGreaterThanOrEqual(2);
  });

  it("triggers onClusterFocusChange when clicking a cluster button", () => {
    const onFocusChange = vi.fn();
    render(
      <ClusterSummaryList
        clusterSummaries={mockSummaries}
        clusterColorMap={mockColorMap}
        focusedClusterId={null}
        onClusterFocusChange={onFocusChange}
      />
    );

    const cluster1Button = screen.getByLabelText("Focus Speaker 1, 5 recordings");
    fireEvent.click(cluster1Button);

    expect(onFocusChange).toHaveBeenCalledWith("Cluster 1");
  });

  it("toggles focus off when clicking the currently focused cluster", () => {
    const onFocusChange = vi.fn();
    render(
      <ClusterSummaryList
        clusterSummaries={mockSummaries}
        clusterColorMap={mockColorMap}
        focusedClusterId="Cluster 1"
        onClusterFocusChange={onFocusChange}
      />
    );

    const cluster1Button = screen.getByLabelText("Focus Speaker 1, 5 recordings");
    fireEvent.click(cluster1Button);

    expect(onFocusChange).toHaveBeenCalledWith(null);
  });
});

import { describe, it, expect, beforeEach, vi } from "vitest";
import { clusterAssignmentStore, type ClusterAssignmentSnapshot } from "../clusterAssignmentStore";

describe("clusterAssignmentStore", () => {
  beforeEach(() => {
    // Reset store snapshot before each test
    clusterAssignmentStore.publish(null);
  });

  const mockSnapshot: ClusterAssignmentSnapshot = {
    fileId: "rec_001",
    stats: {
      recording_id: "rec_001",
      label: "Speaker 1 Clip 1",
      cluster_id: "Cluster 1",
      silhouette_score: 0.85,
      mean_similarity_to_cluster: 0.92,
      min_similarity_to_cluster: 0.88,
      nearest_recording_id: "rec_002",
      nearest_label: "Speaker 1 Clip 2",
      nearest_similarity: 0.94,
      nearest_in_same_cluster: true,
      cluster_fit_margin: 0.15,
      intra_cluster_spread: 0.05,
    },
    clusterSize: 3,
    modelLabel: "ECAPA-TDNN",
    clusteringDistanceThreshold: 0.65,
    clusteringThresholdVersion: "1.0",
    groundTruthGroup: "spk_1",
    groundTruthAvailable: true,
  };

  it("initializes with a null snapshot", () => {
    expect(clusterAssignmentStore.getSnapshot()).toBeNull();
  });

  it("updates snapshot when published", () => {
    clusterAssignmentStore.publish(mockSnapshot);
    expect(clusterAssignmentStore.getSnapshot()).toEqual(mockSnapshot);
  });

  it("notifies active subscribers on publish", () => {
    const listenerA = vi.fn();
    const listenerB = vi.fn();

    const unsubscribeA = clusterAssignmentStore.subscribe(listenerA);
    const unsubscribeB = clusterAssignmentStore.subscribe(listenerB);

    clusterAssignmentStore.publish(mockSnapshot);

    expect(listenerA).toHaveBeenCalledTimes(1);
    expect(listenerB).toHaveBeenCalledTimes(1);

    unsubscribeA();
    unsubscribeB();
  });

  it("stops notifying after unsubscribed", () => {
    const listener = vi.fn();
    const unsubscribe = clusterAssignmentStore.subscribe(listener);

    clusterAssignmentStore.publish(mockSnapshot);
    expect(listener).toHaveBeenCalledTimes(1);

    unsubscribe();
    clusterAssignmentStore.publish(null);
    expect(listener).toHaveBeenCalledTimes(1); // Not called a second time
    expect(clusterAssignmentStore.getSnapshot()).toBeNull();
  });
});

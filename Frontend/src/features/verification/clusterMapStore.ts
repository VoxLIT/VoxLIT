/**
 * Module-level store for the recording_id → cluster_id map produced by the
 * last successful Batch Analysis run. Published by SpeakerVerificationWorkbench
 * (which owns batchResult) and consumed by TaskWorkbench → AudioDatasetPanel →
 * AudioDataTable to drive the Cluster column without prop-drilling through
 * WorkbenchCenterProps.
 */

let clusterMap: Record<string, string> = {};
const listeners = new Set<() => void>();

export const clusterMapStore = {
  subscribe(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
  },
  getSnapshot(): Record<string, string> {
    return clusterMap;
  },
  publish(next: Record<string, string>): void {
    clusterMap = next;
    listeners.forEach((l) => l());
  },
};

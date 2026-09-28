/**
 * Utility to format cluster identifiers (e.g. "cluster-1", "cluster-2", "Cluster 1")
 * into user-friendly speaker labels (e.g. "Speaker 1", "Speaker 2").
 */
export const formatClusterLabel = (clusterId: string | null | undefined): string => {
  if (!clusterId) return "—";
  const trimmed = clusterId.trim();
  const clusterMatch = /^cluster[-_ ]?(\d+)$/i.exec(trimmed);
  if (clusterMatch) {
    return `Speaker ${clusterMatch[1]}`;
  }
  const speakerMatch = /^speaker[-_ ]?(\d+)$/i.exec(trimmed);
  if (speakerMatch) {
    return `Speaker ${speakerMatch[1]}`;
  }
  return trimmed;
};

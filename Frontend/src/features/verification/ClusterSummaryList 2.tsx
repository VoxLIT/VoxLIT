import { useMemo } from "react";
import { HelpCircle, Layers } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import type { ClusterSummary } from "./batchTypes";

interface ClusterSummaryListProps {
  clusterSummaries: ClusterSummary[];
  clusterColorMap: Record<string, string>;
  focusedClusterId?: string | null;
  onClusterFocusChange?: (clusterId: string | null) => void;
}

const formatScore = (value: number) => value.toFixed(4);

// Cluster colors are either "#rrggbb" hex (the 12-color base palette) or an
// "hsl(h, s%, l%)" string (the golden-angle fallback for cluster index >= 12,
// see clusterColors.ts) -- a light background tint must handle both, since a
// naive string-concat alpha suffix (e.g. `${color}1A`) produces invalid CSS
// for the hsl case.
const withAlpha = (color: string, alpha: number): string => {
  const hexMatch = /^#([0-9a-f]{6})$/i.exec(color);
  if (hexMatch) {
    const hex = hexMatch[1];
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }
  const hslMatch = /^hsl\(\s*([\d.]+)\s*,\s*([\d.]+)%\s*,\s*([\d.]+)%\s*\)$/i.exec(color);
  if (hslMatch) {
    const [, h, s, l] = hslMatch;
    return `hsla(${h}, ${s}%, ${l}%, ${alpha})`;
  }
  return color;
};

export const ClusterSummaryList = ({
  clusterSummaries,
  clusterColorMap,
  focusedClusterId,
  onClusterFocusChange,
}: ClusterSummaryListProps) => {
  const sortedClusters = useMemo(
    () =>
      [...clusterSummaries].sort(
        (a, b) =>
          b.member_count - a.member_count ||
          a.cluster_id.localeCompare(b.cluster_id, undefined, { numeric: true })
      ),
    [clusterSummaries]
  );

  const handleClusterClick = (clusterId: string) => {
    onClusterFocusChange?.(focusedClusterId === clusterId ? null : clusterId);
  };

  return (
    <TooltipProvider>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Layers className="h-4 w-4" /> Predicted clusters
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {sortedClusters.length === 0 && (
            <p className="text-xs text-muted-foreground">No clusters to display.</p>
          )}
          {sortedClusters.map((cluster) => {
            const isFocused = focusedClusterId === cluster.cluster_id;
            const clusterColor = clusterColorMap[cluster.cluster_id];
            return (
              <button
                key={cluster.cluster_id}
                type="button"
                onClick={() => handleClusterClick(cluster.cluster_id)}
                aria-pressed={isFocused}
                aria-label={`Focus ${cluster.cluster_id}, ${cluster.member_count} recordings`}
                className="w-full space-y-1.5 rounded-md border p-2 text-left text-xs transition-colors"
                style={
                  isFocused
                    ? { borderColor: clusterColor, borderWidth: 2, backgroundColor: withAlpha(clusterColor, 0.1) }
                    : undefined
                }
              >
                <div className="flex items-center gap-2">
                  <span
                    className="h-2.5 w-2.5 shrink-0 rounded-full"
                    style={{ backgroundColor: clusterColor }}
                  />
                  <span className="font-medium">{cluster.cluster_id}</span>
                  <span className="text-muted-foreground">
                    {cluster.member_count} recording{cluster.member_count === 1 ? "" : "s"}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-1 text-muted-foreground">
                  <span>
                    Mean intra-cluster similarity:{" "}
                    {cluster.mean_intra_cluster_similarity === null
                      ? "Not applicable"
                      : formatScore(cluster.mean_intra_cluster_similarity)}
                  </span>
                  <span className="flex items-center gap-1">
                    Min intra-cluster similarity:{" "}
                    {cluster.min_intra_cluster_similarity === null
                      ? "Not applicable"
                      : formatScore(cluster.min_intra_cluster_similarity)}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle
                          className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors"
                          onClick={(event) => event.stopPropagation()}
                        />
                      </TooltipTrigger>
                      <TooltipContent className="text-xs space-y-1">
                        <p>The weakest similarity between any two recordings in this cluster — the cluster&apos;s loosest link.</p>
                      </TooltipContent>
                    </Tooltip>
                  </span>
                  <span className="flex items-center gap-1">
                    Mean cluster fit score: {formatScore(cluster.mean_fit_score)}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle
                          className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors"
                          onClick={(event) => event.stopPropagation()}
                        />
                      </TooltipTrigger>
                      <TooltipContent className="text-xs space-y-1">
                        <p>How well this cluster&apos;s recordings fit together on average. Higher means the members are more consistently similar to one another.</p>
                      </TooltipContent>
                    </Tooltip>
                  </span>
                  <span className="flex items-center gap-1">
                    Representative clip: {cluster.representative_label}
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle
                          className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors"
                          onClick={(event) => event.stopPropagation()}
                        />
                      </TooltipTrigger>
                      <TooltipContent className="text-xs space-y-1">
                        <p>The recording used to represent this cluster in summaries — not necessarily the &quot;true&quot; center of the cluster.</p>
                      </TooltipContent>
                    </Tooltip>
                  </span>
                </div>
                <p className="truncate text-muted-foreground">{cluster.member_labels.join(", ")}</p>
              </button>
            );
          })}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

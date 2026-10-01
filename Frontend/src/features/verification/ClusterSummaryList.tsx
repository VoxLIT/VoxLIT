import { useMemo } from "react";
import { HelpCircle, Layers } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import { InfoTooltip } from "./InfoTooltip";
import type { ClusterSummary } from "./batchTypes";
import { formatClusterLabel } from "./formatSpeakerLabel";

interface ClusterSummaryListProps {
  clusterSummaries: ClusterSummary[];
  clusterColorMap: Record<string, string>;
  focusedClusterId?: string | null;
  onClusterFocusChange?: (clusterId: string | null) => void;
  resolveLabel?: (id: string) => string;
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
  resolveLabel,
}: ClusterSummaryListProps) => {
  const resolve = resolveLabel ?? ((id: string) => id);
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
            <Layers className="h-4 w-4" /> Predicted speakers
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          {sortedClusters.length === 0 && (
            <p className="text-xs text-muted-foreground">No speakers to display.</p>
          )}
          {sortedClusters.map((cluster) => {
            const isFocused = focusedClusterId === cluster.cluster_id;
            const clusterColor = clusterColorMap[cluster.cluster_id];
            const speakerLabel = formatClusterLabel(cluster.cluster_id);
            return (
              <button
                key={cluster.cluster_id}
                type="button"
                onClick={() => handleClusterClick(cluster.cluster_id)}
                aria-pressed={isFocused}
                aria-label={`Focus ${speakerLabel}, ${cluster.member_count} recordings`}
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
                  <span className="font-medium">{speakerLabel}</span>
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
                    <InfoTooltip title="Lowest similarity in cluster" stopClickPropagation>
                      <p>
                        The weakest similarity between any two recordings in this cluster. It is the cluster&apos;s
                        loosest link.
                      </p>
                    </InfoTooltip>
                  </span>
                  <span className="flex items-center gap-1">
                    Mean cluster fit score: {formatScore(cluster.mean_fit_score)}
                    <InfoTooltip title="Mean cluster fit score" stopClickPropagation>
                      <p>
                        How well this cluster&apos;s recordings fit together on average. Higher means the members are
                        more consistently similar to one another.
                      </p>
                    </InfoTooltip>
                  </span>
                  <span className="flex items-center gap-1">
                    Representative clip: {resolve(cluster.representative_label)}
                    <InfoTooltip title="Representative clip" stopClickPropagation>
                      <p>
                        The recording used to represent this cluster in summaries. It is not necessarily the
                        &quot;true&quot; center of the cluster.
                      </p>
                    </InfoTooltip>
                  </span>
                </div>
                <div
                  className="flex items-center gap-1.5 pt-1 text-muted-foreground text-xs"
                  onClick={(event) => event.stopPropagation()}
                >
                  <span className="font-medium text-foreground shrink-0">Clips ({cluster.member_count}):</span>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <span className="truncate flex-1 font-mono cursor-pointer hover:text-foreground transition-colors inline-flex items-center gap-1">
                        <span className="truncate">{cluster.member_labels.map(resolve).join(", ")}</span>
                        <HelpCircle className="h-3 w-3 shrink-0 text-muted-foreground hover:text-primary cursor-help" />
                      </span>
                    </TooltipTrigger>
                    <TooltipContent
                      side="top"
                      align="start"
                      className="max-w-xs max-h-60 overflow-y-auto p-2.5 text-xs bg-popover text-popover-foreground border shadow-md space-y-1.5"
                    >
                      <div className="font-semibold text-xs border-b pb-1 flex items-center justify-between gap-4">
                        <span>{speakerLabel} Clips</span>
                        <span className="text-muted-foreground font-normal">{cluster.member_count} total</span>
                      </div>
                      <div className="space-y-0.5 font-mono text-[11px] max-h-48 overflow-y-auto pr-1">
                        {cluster.member_labels.map(resolve).map((label, idx) => (
                          <div key={idx} className="truncate py-0.5 border-b border-muted/30 last:border-0 hover:text-primary">
                            {idx + 1}. {label}
                          </div>
                        ))}
                      </div>
                    </TooltipContent>
                  </Tooltip>
                </div>
              </button>
            );
          })}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

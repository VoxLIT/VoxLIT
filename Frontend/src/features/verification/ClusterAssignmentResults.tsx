import { useSyncExternalStore } from "react";
import { HelpCircle, Target } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import type { PredictionResultsProps } from "@/tasks/types";
import { clusterAssignmentStore } from "./clusterAssignmentStore";
import { formatClusterLabel } from "./formatSpeakerLabel";

const formatScore = (value: number) => value.toFixed(4);

/**
 * SV-FR-23 "Cluster Assignment Results" per-clip card, registered as the
 * verification task's PredictionResults slot (see registry.tsx). Its data
 * comes from clusterAssignmentStore, not from these props -- the frozen
 * PredictionResultsProps contract carries no batch/cluster fields, so
 * BatchAnalysisPanel (which owns the batch result) publishes into the store
 * instead. `selectedFile` here only confirms the published snapshot still
 * belongs to the current selection.
 */
export const ClusterAssignmentResults = ({ selectedFile }: PredictionResultsProps) => {
  const snapshot = useSyncExternalStore(
    clusterAssignmentStore.subscribe,
    clusterAssignmentStore.getSnapshot
  );

  const cardHeader = (
    <CardHeader>
      <CardTitle className="flex items-center gap-2 text-sm">
        <Target className="h-4 w-4" /> Speaker assignment results
      </CardTitle>
    </CardHeader>
  );

  if (!selectedFile) {
    return (
      <Card>
        {cardHeader}
        <CardContent>
          <p className="text-xs text-muted-foreground">Select a recording to see its speaker assignment.</p>
        </CardContent>
      </Card>
    );
  }

  if (!snapshot || snapshot.fileId !== selectedFile.file_id) {
    return (
      <Card>
        {cardHeader}
        <CardContent>
          <p className="text-xs text-muted-foreground">
            {snapshot
              ? "This recording is not part of the current batch results."
              : "Run a batch analysis to see speaker assignment results."}
          </p>
        </CardContent>
      </Card>
    );
  }

  const { stats, clusterSize, modelLabel, clusteringDistanceThreshold } =
    snapshot;

  return (
    <TooltipProvider>
      <Card>
        {cardHeader}
        <CardContent className="grid grid-cols-2 gap-x-3 gap-y-1.5 text-xs">
          <span className="text-muted-foreground">Predicted speaker</span>
          <Badge variant="secondary" className="w-fit">{formatClusterLabel(stats.cluster_id)}</Badge>

          <span className="text-muted-foreground">Speaker group size</span>
          <span>{clusterSize} recording{clusterSize === 1 ? "" : "s"}</span>

          <span className="flex items-center gap-1 text-muted-foreground">
            Avg. similarity to cluster
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
              </TooltipTrigger>
              <TooltipContent className="text-xs space-y-1">
                <p>The mean similarity between this recording and every other member of its cluster.</p>
              </TooltipContent>
            </Tooltip>
          </span>
          <span>
            {stats.mean_similarity_to_cluster === null
              ? "Not applicable (single-recording cluster)"
              : formatScore(stats.mean_similarity_to_cluster)}
          </span>

          <span className="flex items-center gap-1 text-muted-foreground">
            Min intra-cluster similarity
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
              </TooltipTrigger>
              <TooltipContent className="text-xs space-y-1">
                <p>The weakest similarity between this recording and any other member of its cluster.</p>
              </TooltipContent>
            </Tooltip>
          </span>
          <span>
            {stats.min_similarity_to_cluster === null
              ? "Not applicable (single-recording cluster)"
              : formatScore(stats.min_similarity_to_cluster)}
          </span>

          <span className="text-muted-foreground">Nearest audio clip</span>
          <span className="truncate" title={snapshot.nearestDisplayLabel || stats.nearest_label}>
            {snapshot.nearestDisplayLabel || stats.nearest_label}
            {!stats.nearest_in_same_cluster && (
              <Badge variant="outline" className="ml-1.5 text-[9px]">different cluster</Badge>
            )}
          </span>

          <span className="flex items-center gap-1 text-muted-foreground">
            Nearest-neighbour similarity
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
              </TooltipTrigger>
              <TooltipContent className="text-xs space-y-1">
                <p>The similarity to the single most similar recording in the whole batch, whether or not it's in the same cluster.</p>
              </TooltipContent>
            </Tooltip>
          </span>
          <span>{formatScore(stats.nearest_similarity)}</span>

          <span className="text-muted-foreground">Selected model</span>
          <span>{modelLabel}</span>

          <span className="flex items-center gap-1 text-muted-foreground">
            Clustering distance threshold
            <Tooltip>
              <TooltipTrigger asChild>
                <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
              </TooltipTrigger>
              <TooltipContent className="text-xs space-y-1">
                <p>The distance threshold used to decide when recordings join the same cluster.</p>
                <p>Separately calibrated from the pair-verification threshold — it governs clustering only, not the same/different-speaker call for a pair.</p>
              </TooltipContent>
            </Tooltip>
          </span>
          <span>{formatScore(clusteringDistanceThreshold)}</span>
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

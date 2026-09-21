import { AlertTriangle, CheckCircle2, GitCompareArrows, HelpCircle, XCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger, TooltipProvider } from "@/components/ui/tooltip";
import type { BatchAnalysisResponse } from "./batchTypes";

interface PairComparisonCardProps {
  selectedLabels: string[];
  batchResult: BatchAnalysisResponse;
  labelToIndex: Map<string, number>;
  resolveLabel?: (id: string) => string;
}

const formatScore = (value: number) => value.toFixed(4);
const formatSignedScore = (value: number) => `${value >= 0 ? "+" : ""}${formatScore(value)}`;

// Pair-verification decisions use the calibrated pair threshold; cluster
// membership comes from average-linkage clustering under a separately
// calibrated distance threshold. The two can legitimately disagree.
const computePairDetails = (batchResult: BatchAnalysisResponse, indexA: number, indexB: number) => {
  const similarity = batchResult.similarity_matrix[indexA][indexB];
  const sameSpeakerDecision = batchResult.decision_matrix[indexA][indexB];
  const clusterA = batchResult.recording_cluster_stats[indexA].cluster_id;
  const clusterB = batchResult.recording_cluster_stats[indexB].cluster_id;
  const sameCluster = clusterA === clusterB;
  return {
    similarity,
    margin: similarity - batchResult.threshold,
    sameSpeakerDecision,
    clusterA,
    clusterB,
    sameCluster,
    thresholdsDisagree: sameSpeakerDecision !== sameCluster,
  };
};

export const PairComparisonCard = ({ selectedLabels, batchResult, labelToIndex, resolveLabel }: PairComparisonCardProps) => {
  const resolve = resolveLabel ?? ((id: string) => id);
  const hasPair = selectedLabels.length === 2;
  const indexA = hasPair ? labelToIndex.get(selectedLabels[0]) : undefined;
  const indexB = hasPair ? labelToIndex.get(selectedLabels[1]) : undefined;
  const canCompare = indexA !== undefined && indexB !== undefined;
  const pairDetails =
    indexA !== undefined && indexB !== undefined ? computePairDetails(batchResult, indexA, indexB) : null;

  return (
    <TooltipProvider>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <GitCompareArrows className="h-4 w-4" /> Pair comparison
          </CardTitle>
        </CardHeader>
        <CardContent>
        {!hasPair && (
          <p className="text-xs text-muted-foreground">Select exactly two points in the plot to compare.</p>
        )}
        {hasPair && !canCompare && (
          <p className="text-xs text-muted-foreground">Selection is no longer valid for the current batch.</p>
        )}
        {canCompare && indexA !== undefined && indexB !== undefined && pairDetails && (
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between gap-2">
              <span className="truncate font-medium" title={resolve(batchResult.labels[indexA])}>{resolve(batchResult.labels[indexA])}</span>
              <span className="text-muted-foreground">↔</span>
              <span className="truncate font-medium" title={resolve(batchResult.labels[indexB])}>{resolve(batchResult.labels[indexB])}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1 text-muted-foreground">
                Cosine similarity
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent className="text-xs space-y-1">
                    <p>How alike these two recordings&apos; voice embeddings are, from -1 to 1. Higher means more similar.</p>
                    <p>This is evidence toward the same/different-speaker call, judged against the threshold below — not proof of identity on its own.</p>
                  </TooltipContent>
                </Tooltip>
              </span>
              <Badge variant="secondary">{formatScore(pairDetails.similarity)}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1 text-muted-foreground">
                Pair threshold ({batchResult.model_label})
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent className="text-xs space-y-1">
                    <p>The calibrated similarity cutoff this model uses to decide same speaker vs. different speakers for a pair.</p>
                    <p>Separately calibrated from the clustering distance threshold, so the two can be different values.</p>
                  </TooltipContent>
                </Tooltip>
              </span>
              <Badge variant="secondary">{formatScore(batchResult.threshold)}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="flex items-center gap-1 text-muted-foreground">
                Threshold margin
                <Tooltip>
                  <TooltipTrigger asChild>
                    <HelpCircle className="h-3 w-3 text-muted-foreground hover:text-primary cursor-help transition-colors" />
                  </TooltipTrigger>
                  <TooltipContent className="text-xs space-y-1">
                    <p>How far this pair&apos;s similarity sits from the decision boundary. A margin near zero means the same/different call was close.</p>
                  </TooltipContent>
                </Tooltip>
              </span>
              <Badge
                variant="secondary"
                className={pairDetails.margin >= 0 ? "text-emerald-700" : "text-rose-700"}
              >
                {formatSignedScore(pairDetails.margin)}
              </Badge>
            </div>
            <div className="flex items-center gap-2">
              {pairDetails.sameSpeakerDecision ? (
                <>
                  <CheckCircle2 className="h-4 w-4 text-emerald-700" />
                  <span className="font-medium text-emerald-700">Same speaker</span>
                </>
              ) : (
                <>
                  <XCircle className="h-4 w-4 text-rose-700" />
                  <span className="font-medium text-rose-700">Different speakers</span>
                </>
              )}
            </div>
            <div className="flex items-center justify-between">
              <span className="truncate text-muted-foreground">{batchResult.labels[indexA]}</span>
              <Badge variant="secondary">{pairDetails.clusterA}</Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="truncate text-muted-foreground">{batchResult.labels[indexB]}</span>
              <Badge variant="secondary">{pairDetails.clusterB}</Badge>
            </div>
            <div className="flex items-center gap-2">
              {pairDetails.sameCluster ? (
                <>
                  <CheckCircle2 className="h-4 w-4 text-emerald-700" />
                  <span className="font-medium text-emerald-700">Same cluster</span>
                </>
              ) : (
                <>
                  <XCircle className="h-4 w-4 text-rose-700" />
                  <span className="font-medium text-rose-700">Different clusters</span>
                </>
              )}
            </div>
            {pairDetails.thresholdsDisagree && (
              <div className="flex items-start gap-1.5 text-muted-foreground">
                <AlertTriangle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                <span>
                  Pair verification and clustering use separately calibrated thresholds, so this pair&apos;s
                  decision and cluster membership can disagree.
                </span>
              </div>
            )}
          </div>
        )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

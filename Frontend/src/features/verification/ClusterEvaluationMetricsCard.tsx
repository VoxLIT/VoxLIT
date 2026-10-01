import { Target } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TooltipProvider } from "@/components/ui/tooltip";
import { InfoTooltip } from "./InfoTooltip";
import type { EvaluationMetrics } from "./batchTypes";

interface ClusterEvaluationMetricsCardProps {
  evaluationMetrics: EvaluationMetrics | null;
  groundTruthAvailable: boolean;
  predictedClusterCount: number;
  trueSpeakerCount: number | null;
}

const formatScore = (value: number) => value.toFixed(4);
const formatCount = (value: number) => value.toLocaleString();

export const ClusterEvaluationMetricsCard = ({
  evaluationMetrics,
  groundTruthAvailable,
  predictedClusterCount,
  trueSpeakerCount,
}: ClusterEvaluationMetricsCardProps) => {
  return (
    <TooltipProvider>
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-sm">
            <Target className="h-4 w-4" /> Ground-truth evaluation
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-xs">
          {!groundTruthAvailable || !evaluationMetrics ? (
            <p className="text-muted-foreground">
              Ground-truth speaker groups were not provided for this batch, so clustering-quality
              metrics are not available. Upload recordings with known speaker groupings to see them
              here.
            </p>
          ) : (
            <>
              <div className="space-y-1">
                <span className="flex items-center gap-1 font-medium">
                  Partition agreement
                  <InfoTooltip title="Partition agreement">
                    <p>
                      ARI (Adjusted Rand Index): how well the predicted clusters match the true speaker groups,
                      corrected for chance. 1.0 is a perfect match; 0 is no better than random.
                    </p>
                    <p>
                      NMI (Normalized Mutual Information): how much the predicted clusters and true groups tell you
                      about each other, from 0 to 1. Higher is a closer match.
                    </p>
                    <p>
                      Purity: the share of recordings in each predicted cluster that belong to that cluster&apos;s most
                      common true speaker, from 0 to 1. Higher is better.
                    </p>
                  </InfoTooltip>
                </span>
                <div className="grid grid-cols-3 gap-1 text-muted-foreground">
                  <span>ARI: {formatScore(evaluationMetrics.adjusted_rand_index)}</span>
                  <span>NMI: {formatScore(evaluationMetrics.normalized_mutual_information)}</span>
                  <span>Purity: {formatScore(evaluationMetrics.cluster_purity)}</span>
                </div>
              </div>

              <div className="space-y-1">
                <span className="flex items-center gap-1 font-medium">
                  Pairwise rates
                  <InfoTooltip title="Pairwise rates">
                    <p>
                      Every pair of recordings is treated as a same-speaker or different-speaker call and compared to
                      the true grouping.
                    </p>
                    <p>
                      Precision: of the pairs predicted same-speaker, the share that truly were. Recall: of the truly
                      same-speaker pairs, the share that were caught.
                    </p>
                    <p>
                      F1 balances precision and recall. Accuracy is the overall share of pairs called correctly. All
                      run from 0 to 1; higher is better.
                    </p>
                  </InfoTooltip>
                </span>
                <div className="grid grid-cols-2 gap-1 text-muted-foreground">
                  <span>Precision: {formatScore(evaluationMetrics.pairwise_precision)}</span>
                  <span>Recall: {formatScore(evaluationMetrics.pairwise_recall)}</span>
                  <span>F1 score: {formatScore(evaluationMetrics.pairwise_f1_score)}</span>
                  <span>Accuracy: {formatScore(evaluationMetrics.pairwise_accuracy)}</span>
                </div>
              </div>

              <div className="space-y-1">
                <span className="flex items-center gap-1 font-medium">
                  Pair counts
                  <InfoTooltip title="Pair counts">
                    <p>
                      How many of the batch&apos;s recording pairs fall into each outcome when the predicted clusters are
                      compared to the true speaker groups.
                    </p>
                    <p>
                      True positive/negative: the predicted and true grouping agree. False positive/negative: they
                      disagree.
                    </p>
                  </InfoTooltip>
                </span>
                <div className="grid grid-cols-3 gap-1 text-muted-foreground">
                  <span>Total: {formatCount(evaluationMetrics.total_unique_pairs)}</span>
                  <span>True positive: {formatCount(evaluationMetrics.true_positive_pairs)}</span>
                  <span>True negative: {formatCount(evaluationMetrics.true_negative_pairs)}</span>
                  <span>False positive: {formatCount(evaluationMetrics.false_positive_pairs)}</span>
                  <span>False negative: {formatCount(evaluationMetrics.false_negative_pairs)}</span>
                </div>
              </div>

              <div className="space-y-1">
                <span className="flex items-center gap-1 font-medium">
                  Speaker count
                  <InfoTooltip title="Speaker count">
                    <p>
                      True speaker groups: the number of distinct speakers known from the ground truth (the known
                      correct grouping). Predicted clusters: the number clustering produced.
                    </p>
                    <p>
                      A mismatch means clustering split one speaker across several clusters, or merged different
                      speakers into one.
                    </p>
                  </InfoTooltip>
                </span>
                <div className="grid grid-cols-2 gap-1 text-muted-foreground">
                  <span>True speaker groups: {trueSpeakerCount ?? "Not available"}</span>
                  <span>Predicted clusters: {predictedClusterCount}</span>
                </div>
              </div>

              <p className="text-muted-foreground">
                These metrics compare the predicted clusters to the ground-truth groups for this
                batch only. They describe how well clustering performed on these recordings, not
                the model&apos;s overall or production-level accuracy.
              </p>
            </>
          )}
        </CardContent>
      </Card>
    </TooltipProvider>
  );
};

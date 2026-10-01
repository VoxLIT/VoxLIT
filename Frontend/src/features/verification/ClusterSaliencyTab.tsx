import { useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { SpeakerSaliencyMap } from "./SpeakerSaliencyMap";
import { verificationAudioUrl } from "./audioUrl";
import type { BatchAnalysisResponse } from "./batchTypes";
import {
  DEFAULT_SALIENCY_BAND_COUNT,
  EMPTY_SALIENCY_RESULTS,
  type SaliencyAxis,
  type SaliencyResultsByAxis,
} from "./saliencyTypes";
import type { UploadedFile } from "@/tasks/types";
import { formatClusterLabel } from "./formatSpeakerLabel";

const DEFAULT_SALIENCY_SEGMENT_COUNT = 8;
const isBackendResolvableId = (id: string) =>
  id.startsWith("rec_") || id.startsWith("asset_") || id.startsWith("crec_");
const isAbortError = (caught: unknown) =>
  caught instanceof DOMException && caught.name === "AbortError";

export interface ClusterSaliencyTabProps {
  model: string;
  selectedFile: UploadedFile | null;
  batchResult: BatchAnalysisResponse | null;
  submittedIds: string[];
  clusterColorMap: Record<string, string>;
}

export const ClusterSaliencyTab = ({
  model,
  selectedFile,
  batchResult,
  submittedIds,
  clusterColorMap,
}: ClusterSaliencyTabProps) => {
  // One result per occlusion axis, so switching axis never refetches.
  const [saliencyAxis, setSaliencyAxis] = useState<SaliencyAxis>("time");
  const [saliencyResults, setSaliencyResults] = useState<SaliencyResultsByAxis>(EMPTY_SALIENCY_RESULTS);
  const [saliencyError, setSaliencyError] = useState<string | null>(null);
  const [isSaliencyLoading, setIsSaliencyLoading] = useState(false);
  const [saliencySegmentCount, setSaliencySegmentCount] = useState(DEFAULT_SALIENCY_SEGMENT_COUNT);

  const saliencyAbortRef = useRef<AbortController | null>(null);
  const isFirstSegmentCountRender = useRef(true);

  // Cluster membership, resolved entirely client-side from the last batch result
  const targetIndex =
    selectedFile && batchResult ? submittedIds.indexOf(selectedFile.file_id) : -1;
  const targetClusterId: string | null =
    targetIndex >= 0 && batchResult ? batchResult.cluster_labels[targetIndex] : null;
  const hasTargetCluster = targetClusterId !== null && targetClusterId !== undefined;
  const clusterMemberIds =
    hasTargetCluster && batchResult
      ? submittedIds.filter((_, i) => i !== targetIndex && batchResult.cluster_labels[i] === targetClusterId)
      : [];
  const isSingletonCluster = hasTargetCluster && clusterMemberIds.length === 0;
  const isTargetIdResolvable = !!selectedFile && isBackendResolvableId(selectedFile.file_id);

  const saliencyEmptyStateMessage = !hasTargetCluster
    ? null
    : !isTargetIdResolvable
      ? "Cluster saliency needs recordings uploaded via the Speaker Verification upload button, not a raw batch upload."
      : isSingletonCluster
        ? "This recording's predicted cluster has no other members — cluster-based saliency requires at least one other recording in the same cluster."
        : null;

  // Invalidate when selectedFile changes
  useEffect(() => {
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
      saliencyAbortRef.current = null;
    }
    setSaliencyResults(EMPTY_SALIENCY_RESULTS);
    setSaliencyError(null);
  }, [selectedFile?.file_id]);

  // Invalidate the time result when segment-count control changes
  useEffect(() => {
    if (isFirstSegmentCountRender.current) {
      isFirstSegmentCountRender.current = false;
      return;
    }
    setSaliencyResults((prev) => ({ ...prev, time: null }));
    setSaliencyError(null);
  }, [saliencySegmentCount]);

  // Invalidate when batchResult or model changes
  useEffect(() => {
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
      saliencyAbortRef.current = null;
    }
    setSaliencyResults(EMPTY_SALIENCY_RESULTS);
    setSaliencyError(null);
  }, [batchResult, model]);

  const runClusterSaliency = async () => {
    if (!selectedFile || !hasTargetCluster || !targetClusterId || isSingletonCluster || !isTargetIdResolvable) return;
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
    }
    const controller = new AbortController();
    saliencyAbortRef.current = controller;

    const formData = new FormData();
    formData.append("model", model);
    formData.append("reference_type", "cluster");
    clusterMemberIds.forEach((id) => formData.append("reference_recording_ids", id));
    formData.append("target_recording_id", selectedFile.file_id);
    formData.append("cluster_id", targetClusterId);
    formData.append("segment_count", String(saliencySegmentCount));
    const requestAxis = saliencyAxis;
    if (requestAxis === "frequency") {
      formData.append("occlusion_axis", "frequency");
      formData.append("band_count", String(DEFAULT_SALIENCY_BAND_COUNT));
    } else if (requestAxis === "integrated_gradients") {
      formData.append("saliency_method", "integrated_gradients");
    }

    setIsSaliencyLoading(true);
    setSaliencyError(null);
    setSaliencyResults((prev) => ({ ...prev, [requestAxis]: null }));

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/explain/saliency`, {
        method: "POST",
        credentials: "include",
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Saliency map failed (${response.status}).`);
      }
      setSaliencyResults((prev) => ({ ...prev, [requestAxis]: payload }));
    } catch (caught) {
      if (isAbortError(caught)) return;
      setSaliencyError(caught instanceof Error ? caught.message : "Saliency map failed.");
    } finally {
      setIsSaliencyLoading(false);
      if (saliencyAbortRef.current === controller) {
        saliencyAbortRef.current = null;
      }
    }
  };

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-sm">Cluster Saliency</CardTitle>
        </CardHeader>
        <CardContent className="space-y-2 text-xs">
          <p className="text-muted-foreground">
            Temporal occlusion saliency computed against the centroid of other recordings in the same speaker cluster.
            It reveals which temporal audio segments of the selected recording contributed most towards its cluster assignment.
          </p>
          <div>
            <span className="font-medium">Selected recording: </span>
            {selectedFile ? (
              <span className="inline-flex items-center gap-2">
                <span>{selectedFile.filename}</span>
                {hasTargetCluster && targetClusterId && (
                  <Badge
                    variant="outline"
                    style={{
                      borderColor: clusterColorMap[targetClusterId] ?? "#3b82f6",
                      color: clusterColorMap[targetClusterId] ?? "#3b82f6",
                    }}
                    className="text-[10px]"
                  >
                    {formatClusterLabel(targetClusterId)}
                  </Badge>
                )}
              </span>
            ) : (
              <span className="text-muted-foreground">
                none selected — click a row in the Audio Dataset table or a point in the embedding graph.
              </span>
            )}
          </div>
          {!batchResult && (
            <p className="text-amber-600 dark:text-amber-400">
              Batch analysis has not been run yet. Please go to the <strong>Batch Analysis</strong> tab and click <strong>Run batch analysis</strong> to compute speaker clusters first.
            </p>
          )}
          {batchResult && selectedFile && targetIndex === -1 && (
            <p className="text-muted-foreground">
              The selected recording was not included in the last batch analysis run.
            </p>
          )}
        </CardContent>
      </Card>

      {batchResult && selectedFile && hasTargetCluster && (
        <SpeakerSaliencyMap
          title={`Cluster saliency — ${selectedFile.filename}`}
          audioUrl={isTargetIdResolvable ? verificationAudioUrl(selectedFile.file_id) : undefined}
          requireCredentials={true}
          result={saliencyResults[saliencyAxis]}
          isLoading={isSaliencyLoading}
          error={saliencyError}
          staleReason={null}
          emptyStateMessage={saliencyEmptyStateMessage}
          onGenerate={saliencyEmptyStateMessage ? null : runClusterSaliency}
          generateLabel="Generate saliency map"
          segmentCount={saliencySegmentCount}
          onSegmentCountChange={setSaliencySegmentCount}
          occlusionAxis={saliencyAxis}
          onOcclusionAxisChange={setSaliencyAxis}
          clusterBadge={
            targetClusterId ? { label: formatClusterLabel(targetClusterId), color: clusterColorMap[targetClusterId] ?? "#3b82f6" } : null
          }
        />
      )}
    </div>
  );
};

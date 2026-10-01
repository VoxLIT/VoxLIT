import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AlertCircle, CheckCircle2, FileAudio, Loader2, Plus, Trash2, Upload, Users, X, XCircle } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { BatchAnalysisPanel } from "./BatchAnalysisPanel";
import { ClusterSaliencyTab } from "./ClusterSaliencyTab";
import { SpeakerSaliencyMap } from "./SpeakerSaliencyMap";
import { buildClusterColorMap } from "./clusterColors";
import type { BatchAnalysisResponse } from "./batchTypes";
import {
  DEFAULT_SALIENCY_BAND_COUNT,
  EMPTY_SALIENCY_RESULTS,
  type AnySaliencyMapResponse,
  type SaliencyAxis,
  type SaliencyResultsByAxis,
} from "./saliencyTypes";
import { PerturbationTools, type VerificationPerturbationContext } from "@/components/analysis/PerturbationTools";
import { clusterMapStore } from "./clusterMapStore";
import { useEmbedding } from "@/contexts/EmbeddingContext";
import type { LocalFilePreview, WorkbenchCenterProps } from "@/tasks/types";

const DEFAULT_SALIENCY_SEGMENT_COUNT = 8;

interface RequestKey {
  model: string;
  enrollmentKey: string;
  probeId: string;
}

type SaliencyRequestKey = RequestKey & { segmentCount: number };

// One entry per occlusion axis, so switching axis never refetches.
const EMPTY_SALIENCY_KEYS: Record<SaliencyAxis, SaliencyRequestKey | null> = { time: null, frequency: null };

function keysEqual(a: RequestKey | null, b: RequestKey | null): boolean {
  if (a === null || a === undefined || b === null || b === undefined) return false;
  return a.model === b.model && a.enrollmentKey === b.enrollmentKey && a.probeId === b.probeId;
}

/** Stable client-only signature for a browser File, used to detect
 *  duplicates/overlaps without touching file content. */
function fileSignature(file: File): string {
  return `${file.name}|${file.size}|${file.type}|${file.lastModified}`;
}

const formatFileSize = (bytes: number): string => {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
};

type SpeakerVerificationWorkbenchProps = WorkbenchCenterProps;

interface ReferenceScore {
  reference_index: number;
  similarity: number;
}

interface VerificationResult {
  model: string;
  model_label: string;
  embedding_dimension: number;
  enrollment_count: number;
  similarity: number;
  threshold: number;
  decision_margin: number;
  same_speaker: boolean;
  enrollment_compactness: number;
  per_reference_scores: ReferenceScore[];
  enrollment_embeddings?: number[][];
  enrollment_centroid: number[];
  probe_embedding: number[];
  calibration: {
    criterion: string;
    dataset: string;
    threshold_locked_before_test_evaluation: boolean;
  };
}

const formatScore = (value: number) => value.toFixed(4);

export const SpeakerVerificationWorkbench = ({
  model,
  modelLabel,
  dataset,
  originalDataset,
  uploadedRawFiles,
  selectedFile,
  selectedEmbeddingFile,
  selectedBatchIds,
  onSelectedBatchIdsChange,
  pairSelection,
  onClearPairSelection,
  datasetRecordings,
  onReprojectHandlerChange,
  onLabelResolverChange,
  onLocalPreviewResolverChange,
  onVerificationAssetCreated,
  localPreview,
  onLocalFileSelect,
}: SpeakerVerificationWorkbenchProps) => {
  const { setEmbeddingDataDirect } = useEmbedding();
  const [activeTab, setActiveTab] = useState("pair-verification");
  const [enrollmentRefs, setEnrollmentRefs] = useState<LocalFilePreview[]>([]);
  const [probeRef, setProbeRef] = useState<LocalFilePreview | null>(null);
  const [result, setResult] = useState<VerificationResult | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const enrollmentInputRef = useRef<HTMLInputElement | null>(null);
  const probeInputRef = useRef<HTMLInputElement | null>(null);

  // Saliency
  const [saliencyAxis, setSaliencyAxis] = useState<SaliencyAxis>("time");
  const [saliencyResults, setSaliencyResults] = useState<SaliencyResultsByAxis>(EMPTY_SALIENCY_RESULTS);
  const [saliencyError, setSaliencyError] = useState<string | null>(null);
  const [isSaliencyLoading, setIsSaliencyLoading] = useState(false);
  const [saliencySegmentCount, setSaliencySegmentCount] = useState(DEFAULT_SALIENCY_SEGMENT_COUNT);
  const [resultGeneratedFor, setResultGeneratedFor] = useState<RequestKey | null>(null);
  const [saliencyGeneratedFor, setSaliencyGeneratedFor] = useState(EMPTY_SALIENCY_KEYS);
  const saliencyResult: AnySaliencyMapResponse | null = saliencyResults[saliencyAxis];

  // Batch analysis context — forwarded to ClusterSaliencyTab
  const [batchResult, setBatchResult] = useState<BatchAnalysisResponse | null>(null);
  const [batchSubmittedIds, setBatchSubmittedIds] = useState<string[]>([]);
  const clusterColorMap = useMemo(
    () => (batchResult ? buildClusterColorMap(batchResult.cluster_labels) : {}),
    [batchResult]
  );

  const handleBatchResultChange = useCallback(
    (nextResult: BatchAnalysisResponse | null, nextSubmittedIds: string[]) => {
      setBatchResult(nextResult);
      setBatchSubmittedIds(nextSubmittedIds);
    },
    []
  );

  // Publish recording_id → cluster_id map so the Audio Dataset table can
  // show cluster assignments in its Cluster column without prop drilling.
  useEffect(() => {
    if (!batchResult) {
      clusterMapStore.publish({});
      return;
    }
    const map: Record<string, string> = {};
    batchSubmittedIds.forEach((id, i) => {
      if (batchResult.cluster_labels[i] !== undefined) {
        map[id] = batchResult.cluster_labels[i];
      }
    });
    clusterMapStore.publish(map);
  }, [batchResult, batchSubmittedIds]);

  const deviceVerifyAbortRef = useRef<AbortController | null>(null);
  const deviceVerifyKeyRef = useRef<RequestKey | null>(null);
  const saliencyAbortRef = useRef<AbortController | null>(null);
  const saliencyKeyRef = useRef<SaliencyRequestKey | null>(null);

  // Resolve an opaque recording_id to a display label for Perturbation
  const resolveRecordingLabel = (id: string): string =>
    datasetRecordings?.find((r) => r.recording_id === id)?.display_filename ?? id;

  // External audio verification requirements: 3-5 enrollment clips, 1 probe clip
  const canRunDevice = enrollmentRefs.length >= 3 && enrollmentRefs.length <= 5 && !!probeRef && !!model;
  const deviceEnrollmentKey = [...enrollmentRefs.map((r) => r.localId)].sort().join(",");
  const currentDeviceKey: RequestKey | null =
    probeRef && enrollmentRefs.length >= 3 && enrollmentRefs.length <= 5
      ? { model, enrollmentKey: deviceEnrollmentKey, probeId: probeRef.localId }
      : null;

  const currentDeviceKeyRef = useRef<RequestKey | null>(null);
  currentDeviceKeyRef.current = currentDeviceKey;

  const isResultCurrent =
    result !== null &&
    keysEqual(resultGeneratedFor, currentDeviceKey);
  const isSaliencyStale =
    saliencyResult !== null &&
    (!keysEqual(saliencyGeneratedFor[saliencyAxis], currentDeviceKey) ||
      // Segment count only affects the time view.
      (saliencyAxis === "time" && saliencyGeneratedFor.time?.segmentCount !== saliencySegmentCount));

  useEffect(() => {
    if (deviceVerifyAbortRef.current && !keysEqual(deviceVerifyKeyRef.current, currentDeviceKey)) {
      deviceVerifyAbortRef.current.abort();
    }
    const sk = saliencyKeyRef.current;
    if (sk && saliencyAbortRef.current && !keysEqual(sk, currentDeviceKey)) {
      saliencyAbortRef.current.abort();
    }
  }, [currentDeviceKey?.model, currentDeviceKey?.enrollmentKey, currentDeviceKey?.probeId]);

  // Perturbation tab context — single currently selected recording, driven
  // by the same shared selectedFile the rest of the app uses.
  const verificationPerturbationContext: VerificationPerturbationContext = {
    model,
    selectedRecordingId: selectedFile?.file_id ?? null,
    selectedRecordingLabel: selectedFile?.file_id ? resolveRecordingLabel(selectedFile.file_id) : null,
    onPerturbationApplied: (res) => onVerificationAssetCreated(res.session_asset),
  };

  useEffect(() => {
    setResult(null);
    setResultGeneratedFor(null);
    setError(null);
    setSaliencyResults(EMPTY_SALIENCY_RESULTS);
    setSaliencyError(null);
    setSaliencyGeneratedFor(EMPTY_SALIENCY_KEYS);
    saliencyKeyRef.current = null;
  }, [model]);

  // Revoke each accepted file's preview URL when it's replaced or the
  // component unmounts.
  useEffect(() => {
    return () => {
      enrollmentRefs.forEach((ref) => URL.revokeObjectURL(ref.previewUrl));
    };
  }, [enrollmentRefs]);

  useEffect(() => {
    return () => {
      if (probeRef) URL.revokeObjectURL(probeRef.previewUrl);
    };
  }, [probeRef]);

  const scoreProgress = useMemo(
    () => (result ? Math.max(0, Math.min(100, ((result.similarity + 1) / 2) * 100)) : 0),
    [result]
  );

  const handleEnrollmentChange = (event: ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files ?? []);
    if (files.length === 0) return;
    event.target.value = ""; // Reset input so re-selecting the same file fires onChange

    const messages: string[] = [];
    const currentSignatures = new Set(enrollmentRefs.map((r) => fileSignature(r.file)));
    const probeSignature = probeRef ? fileSignature(probeRef.file) : null;

    const toAdd: File[] = [];
    let duplicateCount = 0;
    let probeOverlapCount = 0;

    for (const file of files) {
      const sig = fileSignature(file);
      if (currentSignatures.has(sig)) {
        duplicateCount += 1;
        continue;
      }
      if (probeSignature && sig === probeSignature) {
        probeOverlapCount += 1;
        continue;
      }
      currentSignatures.add(sig);
      toAdd.push(file);
    }

    if (duplicateCount > 0) {
      messages.push(`${duplicateCount} duplicate clip${duplicateCount > 1 ? "s" : ""} ignored.`);
    }
    if (probeOverlapCount > 0) {
      messages.push("Clips matching the probe recording cannot be used as enrolment references.");
    }

    const availableSlots = 5 - enrollmentRefs.length;
    let accepted = toAdd;
    if (accepted.length > availableSlots) {
      accepted = accepted.slice(0, availableSlots);
      messages.push(`Only ${availableSlots} more clip${availableSlots > 1 ? "s" : ""} could be added (max 5).`);
    }

    if (accepted.length > 0) {
      const newRefs: LocalFilePreview[] = accepted.map((file, idx) => ({
        localId: crypto.randomUUID(),
        file,
        previewUrl: URL.createObjectURL(file),
        role: "enrollment" as const,
        label: `Reference ${enrollmentRefs.length + idx + 1}`,
      }));
      setEnrollmentRefs((prev) => [...prev, ...newRefs]);
      setResult(null);
      setResultGeneratedFor(null);
      setSaliencyResults(EMPTY_SALIENCY_RESULTS);
      setSaliencyGeneratedFor(EMPTY_SALIENCY_KEYS);
    }

    setError(messages.length > 0 ? messages.join(" ") : null);
  };

  const handleRemoveEnrollment = (localId: string) => {
    setEnrollmentRefs((prev) => {
      const target = prev.find((r) => r.localId === localId);
      if (target) {
        URL.revokeObjectURL(target.previewUrl);
        if (localPreview?.localId === localId) {
          onLocalFileSelect(null);
        }
      }
      const updated = prev.filter((r) => r.localId !== localId);
      return updated.map((r, i) => ({ ...r, label: `Reference ${i + 1}` }));
    });
    setResult(null);
    setResultGeneratedFor(null);
    setSaliencyResults(EMPTY_SALIENCY_RESULTS);
    setSaliencyGeneratedFor(EMPTY_SALIENCY_KEYS);
  };

  const handleClearEnrollment = () => {
    enrollmentRefs.forEach((r) => URL.revokeObjectURL(r.previewUrl));
    if (localPreview?.role === "enrollment") {
      onLocalFileSelect(null);
    }
    setEnrollmentRefs([]);
    setResult(null);
    setResultGeneratedFor(null);
    setSaliencyResults(EMPTY_SALIENCY_RESULTS);
    setSaliencyGeneratedFor(EMPTY_SALIENCY_KEYS);
  };

  const handleProbeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;
    event.target.value = "";
    if (!file) return;

    const signature = fileSignature(file);
    const conflict = enrollmentRefs.some((ref) => fileSignature(ref.file) === signature);
    if (conflict) {
      setError("This recording is already selected as an enrolment reference.");
      return;
    }

    if (probeRef) {
      URL.revokeObjectURL(probeRef.previewUrl);
      if (localPreview?.role === "probe") {
        onLocalFileSelect(null);
      }
    }

    setProbeRef({
      localId: crypto.randomUUID(),
      file,
      previewUrl: URL.createObjectURL(file),
      role: "probe" as const,
      label: "Probe",
    });

    setResult(null);
    setResultGeneratedFor(null);
    setSaliencyResults(EMPTY_SALIENCY_RESULTS);
    setSaliencyGeneratedFor(EMPTY_SALIENCY_KEYS);
    setError(null);
  };

  const handleRemoveProbe = () => {
    if (probeRef) {
      URL.revokeObjectURL(probeRef.previewUrl);
      if (localPreview?.localId === probeRef.localId) {
        onLocalFileSelect(null);
      }
    }
    setProbeRef(null);
    setResult(null);
    setResultGeneratedFor(null);
    setSaliencyResults(EMPTY_SALIENCY_RESULTS);
    setSaliencyGeneratedFor(EMPTY_SALIENCY_KEYS);
  };

  const projectPairEmbeddings = useCallback(
    async (
      verifyResult: VerificationResult,
      currentEnrollments: LocalFilePreview[],
      currentProbe: LocalFilePreview,
      method: string = "pca",
      nComponents: 2 | 3 = 2
    ) => {
      if (!verifyResult.enrollment_embeddings || verifyResult.enrollment_embeddings.length === 0) {
        return;
      }
      const allEmbeddings = [...verifyResult.enrollment_embeddings, verifyResult.probe_embedding];
      const allLabels = [
        ...currentEnrollments.map((r, i) => `Reference ${i + 1}: ${r.file.name}`),
        `Probe: ${currentProbe.file.name}`,
      ];

      try {
        const response = await fetch(`${API_BASE}/tasks/verification/batch/project`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            model: verifyResult.model,
            embeddings: allEmbeddings,
            labels: allLabels,
            reduction_method: method,
            n_components: nComponents,
          }),
        });

        if (!response.ok) return;
        const projPayload = await response.json();

        setEmbeddingDataDirect({
          model: verifyResult.model,
          dataset: "custom:pair-verification",
          reduction_method: method,
          n_components: nComponents,
          embeddings: [],
          total_files: allLabels.length,
          original_dimension: verifyResult.embedding_dimension,
          reduction_method_used: projPayload.reduction_method_used,
          effective_components: projPayload.effective_components,
          reduced_embeddings: [
            ...currentEnrollments.map((ref, i) => ({
              filename: ref.file.name,
              displayFilename: `${ref.label} (${ref.file.name})`,
              coordinates: projPayload.coordinates[i],
              color: "#2563eb",
              hoverExtra: `Enrolment Reference ${i + 1} • Similarity to probe: ${formatScore(verifyResult.per_reference_scores[i]?.similarity ?? 0)}`,
              clusterId: "Reference Speaker",
            })),
            {
              filename: currentProbe.file.name,
              displayFilename: `Probe (${currentProbe.file.name})`,
              coordinates: projPayload.coordinates[currentEnrollments.length],
              color: verifyResult.same_speaker ? "#16a34a" : "#dc2626",
              hoverExtra: `Probe Clip • Similarity to centroid: ${formatScore(verifyResult.similarity)} (${verifyResult.same_speaker ? "Same speaker" : "Different speakers"})`,
              clusterId: verifyResult.same_speaker ? "Reference Speaker" : "Different Speaker",
            },
          ],
        });
      } catch (err) {
        console.error("Failed to project pair verification embeddings:", err);
      }
    },
    [setEmbeddingDataDirect]
  );

  const runVerification = async () => {
    if (!canRunDevice || !probeRef || !currentDeviceKey) return;

    const requestKey = currentDeviceKey;
    if (deviceVerifyAbortRef.current) {
      deviceVerifyAbortRef.current.abort();
    }
    const controller = new AbortController();
    deviceVerifyAbortRef.current = controller;
    deviceVerifyKeyRef.current = requestKey;

    const formData = new FormData();
    formData.append("model", model);
    enrollmentRefs.forEach((ref) => formData.append("enrollment_files", ref.file));
    formData.append("probe_file", probeRef.file);

    setIsRunning(true);
    setError(null);
    setResult(null);
    setResultGeneratedFor(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/verify`, {
        method: "POST",
        credentials: "include",
        body: formData,
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Verification failed (${response.status})`);
      }
      if (!keysEqual(requestKey, currentDeviceKeyRef.current)) return;
      const verifyResult = payload as VerificationResult;
      setResult(verifyResult);
      setResultGeneratedFor(requestKey);
      if (saliencyGeneratedFor.time || saliencyGeneratedFor.frequency) {
        setSaliencyResults(EMPTY_SALIENCY_RESULTS);
        setSaliencyError(null);
        setSaliencyGeneratedFor(EMPTY_SALIENCY_KEYS);
        saliencyKeyRef.current = null;
      }

      // Add the (3 to 5 enrollment + 1 probe) clips to the graph
      await projectPairEmbeddings(verifyResult, enrollmentRefs, probeRef, "pca", 2);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setError(caught instanceof Error ? caught.message : "Speaker verification failed.");
    } finally {
      setIsRunning(false);
      if (deviceVerifyAbortRef.current === controller) {
        deviceVerifyAbortRef.current = null;
      }
    }
  };

  const handlePairReproject = useCallback(
    (method: string, n: number) => {
      if (result && probeRef && enrollmentRefs.length >= 3) {
        projectPairEmbeddings(result, enrollmentRefs, probeRef, method, n === 3 ? 3 : 2);
      }
    },
    [result, probeRef, enrollmentRefs, projectPairEmbeddings]
  );

  useEffect(() => {
    if (activeTab === "pair-verification" && result && probeRef && enrollmentRefs.length >= 3) {
      onReprojectHandlerChange(handlePairReproject);
    }
  }, [activeTab, result, probeRef, enrollmentRefs.length, handlePairReproject, onReprojectHandlerChange]);

  useEffect(() => {
    if (activeTab === "pair-verification") {
      const resolver = (label: string): LocalFilePreview | null => {
        const foundRef = enrollmentRefs.find(
          (r) => r.file.name === label || r.localId === label || r.label === label
        );
        if (foundRef) return foundRef;
        if (probeRef && (probeRef.file.name === label || probeRef.localId === label || probeRef.label === label)) {
          return probeRef;
        }
        return null;
      };
      onLocalPreviewResolverChange?.(resolver);
    } else {
      onLocalPreviewResolverChange?.(null);
    }
    return () => {
      onLocalPreviewResolverChange?.(null);
    };
  }, [activeTab, enrollmentRefs, probeRef, onLocalPreviewResolverChange]);

  useEffect(() => {
    if (activeTab !== "pair-verification" || !selectedEmbeddingFile) return;
    const match =
      enrollmentRefs.find((r) => r.file.name === selectedEmbeddingFile || r.localId === selectedEmbeddingFile) ||
      (probeRef && (probeRef.file.name === selectedEmbeddingFile || probeRef.localId === selectedEmbeddingFile) ? probeRef : null);
    if (match && localPreview?.localId !== match.localId) {
      onLocalFileSelect(match);
    }
  }, [activeTab, selectedEmbeddingFile, enrollmentRefs, probeRef, localPreview, onLocalFileSelect]);

  const runUploadSaliency = async () => {
    if (!isResultCurrent || !probeRef || !currentDeviceKey) return;
    if (saliencyAbortRef.current) {
      saliencyAbortRef.current.abort();
    }
    const controller = new AbortController();
    saliencyAbortRef.current = controller;
    const requestKey = { ...currentDeviceKey, segmentCount: saliencySegmentCount };
    saliencyKeyRef.current = requestKey;
    const requestAxis = saliencyAxis;

    const formData = new FormData();
    formData.append("model", model);
    formData.append("reference_type", "enrollment");
    enrollmentRefs.forEach((ref) => formData.append("enrollment_files", ref.file));
    formData.append("probe_file", probeRef.file);
    formData.append("segment_count", String(saliencySegmentCount));
    if (requestAxis === "frequency") {
      formData.append("occlusion_axis", "frequency");
      formData.append("band_count", String(DEFAULT_SALIENCY_BAND_COUNT));
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
      if (!keysEqual(requestKey, currentDeviceKeyRef.current)) return;
      setSaliencyResults((prev) => ({ ...prev, [requestAxis]: payload }));
      setSaliencyGeneratedFor((prev) => ({ ...prev, [requestAxis]: requestKey }));
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setSaliencyError(caught instanceof Error ? caught.message : "Saliency map failed.");
    } finally {
      setIsSaliencyLoading(false);
      if (saliencyAbortRef.current === controller) {
        saliencyAbortRef.current = null;
      }
    }
  };

  return (
    <Tabs defaultValue="pair-verification" value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
      <div className="flex-shrink-0 bg-panel-header border-b border-border px-3 py-2">
        <TabsList className="h-7 grid w-full grid-cols-4 bg-muted">
          <TabsTrigger value="pair-verification" className="text-xs">Pair Verification</TabsTrigger>
          <TabsTrigger value="batch-analysis" className="text-xs">Batch Analysis</TabsTrigger>
          <TabsTrigger value="cluster-saliency" className="text-xs">Cluster Saliency</TabsTrigger>
          <TabsTrigger value="perturbation" className="text-xs">Perturbation</TabsTrigger>
        </TabsList>
      </div>

      <div className="flex-1 overflow-y-auto bg-background p-4 scrollbar-thin">
        <div className="mx-auto max-w-4xl space-y-4">
          <TabsContent value="pair-verification" className="space-y-4">
            <div className="rounded-lg border border-border/60 bg-muted/30 p-3.5">
              <h3 className="text-sm font-medium">External Speaker Pair Verification</h3>
              <p className="text-xs text-muted-foreground mt-0.5">
                Verify speaker identity using external audio clips. Upload <strong>3 to 5</strong> enrolment clips from a known reference speaker, plus <strong>1</strong> probe clip to compare.
              </p>
            </div>

            <div className="grid gap-4 md:grid-cols-2">
              {/* Card 1: Enrolment recordings */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-sm font-semibold">
                      <Users className="h-4 w-4 text-primary" /> 1. Enrolment recordings
                    </CardTitle>
                    {enrollmentRefs.length === 0 ? (
                      <Badge variant="outline" className="text-xs font-normal">0/5 (3–5 required)</Badge>
                    ) : enrollmentRefs.length < 3 ? (
                      <Badge variant="outline" className="border-amber-400 text-amber-600 bg-amber-50 text-xs">
                        {enrollmentRefs.length}/5 ({3 - enrollmentRefs.length} more needed)
                      </Badge>
                    ) : (
                      <Badge variant="default" className="bg-emerald-600 hover:bg-emerald-700 text-xs">
                        {enrollmentRefs.length}/5 (Ready)
                      </Badge>
                    )}
                  </div>
                  <CardDescription className="text-xs">
                    Clips of the reference speaker to construct the enrolment profile.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <input
                    ref={enrollmentInputRef}
                    type="file"
                    accept="audio/wav,audio/mpeg,audio/mp4,audio/flac,.wav,.mp3,.m4a,.flac"
                    multiple
                    onChange={handleEnrollmentChange}
                    className="hidden"
                  />

                  {enrollmentRefs.length === 0 ? (
                    <div
                      onClick={() => enrollmentInputRef.current?.click()}
                      className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-muted-foreground/25 rounded-lg cursor-pointer hover:border-primary/50 hover:bg-muted/40 transition-colors"
                    >
                      <Upload className="h-7 w-7 text-muted-foreground/60 mb-2" />
                      <p className="text-xs font-medium text-foreground">Click to upload enrolment clips</p>
                      <p className="text-[11px] text-muted-foreground mt-0.5">Select 3 to 5 audio clips (WAV, MP3, FLAC, M4A)</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div className="space-y-1.5 max-h-56 overflow-y-auto pr-1">
                        {enrollmentRefs.map((ref) => {
                          const isSelected = localPreview?.localId === ref.localId;
                          return (
                            <div
                              key={ref.localId}
                              className={`flex items-center justify-between gap-2 rounded-md border px-2.5 py-1.5 text-xs transition-colors ${
                                isSelected
                                  ? "border-primary bg-primary/10 text-primary"
                                  : "border-border hover:bg-muted/60"
                              }`}
                            >
                              <button
                                type="button"
                                onClick={() => onLocalFileSelect(ref)}
                                className="flex items-center gap-2 min-w-0 flex-1 text-left"
                              >
                                <FileAudio className="h-4 w-4 shrink-0 text-muted-foreground" />
                                <div className="min-w-0 flex-1">
                                  <div className="flex items-center gap-1.5">
                                    <span className="font-medium text-xs truncate">{ref.label}</span>
                                    <span className="text-[11px] text-muted-foreground">({formatFileSize(ref.file.size)})</span>
                                  </div>
                                  <p className="text-[11px] text-muted-foreground truncate">{ref.file.name}</p>
                                </div>
                              </button>
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => handleRemoveEnrollment(ref.localId)}
                                className="h-6 w-6 text-muted-foreground hover:text-destructive shrink-0"
                                title="Remove clip"
                              >
                                <X className="h-3.5 w-3.5" />
                              </Button>
                            </div>
                          );
                        })}
                      </div>

                      <div className="flex items-center justify-between pt-1">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => enrollmentInputRef.current?.click()}
                          disabled={enrollmentRefs.length >= 5}
                          className="text-xs h-7 gap-1"
                        >
                          <Plus className="h-3.5 w-3.5" /> Add more clips
                        </Button>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={handleClearEnrollment}
                          className="text-xs h-7 text-muted-foreground hover:text-destructive gap-1"
                        >
                          <Trash2 className="h-3.5 w-3.5" /> Clear all
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>

              {/* Card 2: Probe recording */}
              <Card>
                <CardHeader className="pb-3">
                  <div className="flex items-center justify-between">
                    <CardTitle className="flex items-center gap-2 text-sm font-semibold">
                      <FileAudio className="h-4 w-4 text-primary" /> 2. Probe recording
                    </CardTitle>
                    {probeRef ? (
                      <Badge variant="default" className="bg-emerald-600 hover:bg-emerald-700 text-xs">Ready</Badge>
                    ) : (
                      <Badge variant="outline" className="text-xs font-normal">1 required</Badge>
                    )}
                  </div>
                  <CardDescription className="text-xs">
                    Recording whose identity should be verified against the enrolment profile.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <input
                    ref={probeInputRef}
                    type="file"
                    accept="audio/wav,audio/mpeg,audio/mp4,audio/flac,.wav,.mp3,.m4a,.flac"
                    onChange={handleProbeChange}
                    className="hidden"
                  />

                  {!probeRef ? (
                    <div
                      onClick={() => probeInputRef.current?.click()}
                      className="flex flex-col items-center justify-center p-6 border-2 border-dashed border-muted-foreground/25 rounded-lg cursor-pointer hover:border-primary/50 hover:bg-muted/40 transition-colors"
                    >
                      <Upload className="h-7 w-7 text-muted-foreground/60 mb-2" />
                      <p className="text-xs font-medium text-foreground">Click to upload probe recording</p>
                      <p className="text-[11px] text-muted-foreground mt-0.5">Select exactly 1 audio clip (WAV, MP3, FLAC, M4A)</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <div
                        className={`flex items-center justify-between gap-2 rounded-md border px-2.5 py-2 text-xs transition-colors ${
                          localPreview?.localId === probeRef.localId
                            ? "border-primary bg-primary/10 text-primary"
                            : "border-border hover:bg-muted/60"
                        }`}
                      >
                        <button
                          type="button"
                          onClick={() => onLocalFileSelect(probeRef)}
                          className="flex items-center gap-2 min-w-0 flex-1 text-left"
                        >
                          <FileAudio className="h-4 w-4 shrink-0 text-muted-foreground" />
                          <div className="min-w-0 flex-1">
                            <div className="flex items-center gap-1.5">
                              <span className="font-medium text-xs">Probe Recording</span>
                              <span className="text-[11px] text-muted-foreground">({formatFileSize(probeRef.file.size)})</span>
                            </div>
                            <p className="text-[11px] text-muted-foreground truncate">{probeRef.file.name}</p>
                          </div>
                        </button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={handleRemoveProbe}
                          className="h-6 w-6 text-muted-foreground hover:text-destructive shrink-0"
                          title="Remove probe"
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                      </div>

                      <div className="pt-1">
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => probeInputRef.current?.click()}
                          className="text-xs h-7 gap-1"
                        >
                          <Upload className="h-3.5 w-3.5" /> Replace probe clip
                        </Button>
                      </div>
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Action Button & Status */}
            <div className="space-y-2">
              <Button
                className="w-full h-10 text-sm font-semibold"
                disabled={!canRunDevice || isRunning}
                onClick={runVerification}
              >
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Extracting embeddings & verifying…
                  </>
                ) : (
                  "Verify speaker"
                )}
              </Button>
              {!canRunDevice && !isRunning && (
                <p className="text-xs text-muted-foreground text-center">
                  {enrollmentRefs.length < 3
                    ? `Upload at least ${3 - enrollmentRefs.length} more enrolment clip${3 - enrollmentRefs.length > 1 ? "s" : ""} (currently ${enrollmentRefs.length}/5) to enable verification.`
                    : !probeRef
                      ? "Upload 1 probe recording to compare against the enrolment profile."
                      : !model
                        ? "Select a speaker verification model from the top toolbar."
                        : null}
                </p>
              )}
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Verification could not be completed</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {result && (
              <div className="space-y-4">
                <Alert className={result.same_speaker ? "border-emerald-300 bg-emerald-50" : "border-rose-300 bg-rose-50"}>
                  {result.same_speaker ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-700" />
                  ) : (
                    <XCircle className="h-4 w-4 text-rose-700" />
                  )}
                  <AlertTitle>{result.same_speaker ? "Same speaker" : "Different speakers"}</AlertTitle>
                  <AlertDescription>
                    Cosine similarity {formatScore(result.similarity)} is {result.same_speaker ? "above" : "below"} the calibrated threshold {formatScore(result.threshold)}.
                  </AlertDescription>
                </Alert>

                <div className="grid gap-4 md:grid-cols-3">
                  <Card>
                    <CardHeader><CardTitle className="text-xs">Similarity score</CardTitle></CardHeader>
                    <CardContent>
                      <div className="mb-2 text-2xl font-semibold">{formatScore(result.similarity)}</div>
                      <Progress value={scoreProgress} />
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader><CardTitle className="text-xs">Decision margin</CardTitle></CardHeader>
                    <CardContent className="text-2xl font-semibold">
                      {result.decision_margin >= 0 ? "+" : ""}{formatScore(result.decision_margin)}
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader><CardTitle className="text-xs">Cluster compactness</CardTitle></CardHeader>
                    <CardContent className="text-2xl font-semibold">
                      {formatScore(result.enrollment_compactness)}
                    </CardContent>
                  </Card>
                </div>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-sm">Pair comparison breakdown</CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-2">
                    {result.per_reference_scores.map((item) => (
                      <div key={item.reference_index} className="flex items-center justify-between text-sm">
                        <span>Reference {item.reference_index} ↔ probe</span>
                        <Badge variant="secondary">{formatScore(item.similarity)}</Badge>
                      </div>
                    ))}
                  </CardContent>
                </Card>

                <SpeakerSaliencyMap
                  title={saliencyAxis === "time" ? "Temporal occlusion saliency" : "Frequency-band occlusion saliency"}
                  audioUrl={probeRef?.previewUrl}
                  requireCredentials={false}
                  result={saliencyResult}
                  isLoading={isSaliencyLoading}
                  error={saliencyError}
                  staleReason={
                    !isResultCurrent
                      ? "Recordings have changed since this verification result was produced — run verification again to enable a saliency map."
                      : isSaliencyStale
                        ? "Segment count changed since this map was generated."
                        : null
                  }
                  emptyStateMessage={null}
                  onGenerate={isResultCurrent ? runUploadSaliency : null}
                  generateLabel="Generate saliency map"
                  segmentCount={saliencySegmentCount}
                  onSegmentCountChange={setSaliencySegmentCount}
                  occlusionAxis={saliencyAxis}
                  onOcclusionAxisChange={setSaliencyAxis}
                />
              </div>
            )}
          </TabsContent>

          <TabsContent value="batch-analysis" forceMount className="space-y-4 data-[state=inactive]:hidden">
            <BatchAnalysisPanel
              model={model}
              modelLabel={modelLabel}
              dataset={dataset}
              originalDataset={originalDataset}
              uploadedRawFiles={uploadedRawFiles}
              selectedBatchIds={selectedBatchIds}
              pairSelection={pairSelection}
              selectedFile={selectedFile}
              datasetRecordings={datasetRecordings}
              onReprojectHandlerChange={onReprojectHandlerChange}
              onLabelResolverChange={onLabelResolverChange}
              onBatchResultChange={handleBatchResultChange}
            />
          </TabsContent>

          <TabsContent value="cluster-saliency" forceMount className="space-y-4 data-[state=inactive]:hidden">
            <ClusterSaliencyTab
              model={model}
              selectedFile={selectedFile}
              batchResult={batchResult}
              submittedIds={batchSubmittedIds}
              clusterColorMap={clusterColorMap}
            />
          </TabsContent>

          <TabsContent value="perturbation" forceMount className="space-y-4 data-[state=inactive]:hidden">
            <PerturbationTools selectedFile={null} verification={verificationPerturbationContext} />
          </TabsContent>
        </div>
      </div>
    </Tabs>
  );
};

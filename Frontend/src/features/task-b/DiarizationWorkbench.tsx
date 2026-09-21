import { useEffect, useRef, useState } from "react";
import { AudioLines, Loader2, Play, Upload } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DiarizationTimeline } from "./DiarizationTimeline";
import { SimilarityMatrix } from "./SimilarityMatrix";
import { EmbeddingScatter } from "./EmbeddingScatter";
import { DeltaSummaryCard } from "./DeltaSummaryCard";
import { PerturbationControls, noisePercentToLevel } from "./PerturbationControls";
import { StackedTimelines } from "./StackedTimelines";
import {
  DiarizationModelInfo,
  DiarizationResult,
  PerturbationResult,
  PerturbationType,
  ProjectionResult,
  RecordingInfo,
} from "./types";

interface DiarizationWorkbenchProps {
  model: string;
  modelLabel: string;
}

/** Session uploads and demo recordings differ only by id prefix; the audio
 *  endpoint differs, everything downstream does not. */
const UPLOAD_ID_PREFIX = "upl_";

const audioUrlFor = (recordingId: string): string =>
  recordingId.startsWith(UPLOAD_ID_PREFIX)
    ? `${API_BASE}/tasks/task-b/uploads/${encodeURIComponent(recordingId)}/audio`
    : `${API_BASE}/tasks/task-b/dataset/recordings/${encodeURIComponent(recordingId)}/audio`;

export const DiarizationWorkbench = ({ model, modelLabel }: DiarizationWorkbenchProps) => {
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [uploads, setUploads] = useState<RecordingInfo[]>([]);
  const [models, setModels] = useState<DiarizationModelInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  const [selectedRecordingId, setSelectedRecordingId] = useState<string>("");
  const [result, setResult] = useState<DiarizationResult | null>(null);
  const [projection, setProjection] = useState<ProjectionResult | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // --- Perturbation counterfactual -------------------------------------
  const [perturbationType, setPerturbationType] = useState<PerturbationType>("noise");
  const [noisePercent, setNoisePercent] = useState(4);
  const [maskRange, setMaskRange] = useState<[number, number]>([20, 40]);
  const [perturbation, setPerturbation] = useState<PerturbationResult | null>(null);
  const [isPerturbing, setIsPerturbing] = useState(false);
  const [perturbationError, setPerturbationError] = useState<string | null>(null);
  // A perturbed run can take minutes, so an abandoned one must not land on
  // top of a newer selection. Same guard the verification workbench uses.
  const perturbationAbortRef = useRef<AbortController | null>(null);
  const perturbedAudioRef = useRef<HTMLAudioElement>(null);

  /** Anything that changes which audio is under test invalidates a perturbed
   *  comparison, so clear it rather than leave a result pointing at a
   *  recording the user has moved away from. */
  const resetPerturbation = () => {
    perturbationAbortRef.current?.abort();
    perturbationAbortRef.current = null;
    setPerturbation(null);
    setPerturbationError(null);
  };

  // Queue for "play segment A, then segment B" from the similarity matrix.
  const playQueueRef = useRef<{ start: number; end: number }[]>([]);

  const handleTimeUpdate = () => {
    const audio = audioRef.current;
    const queue = playQueueRef.current;
    if (!audio || queue.length === 0) return;
    if (audio.currentTime >= queue[0].end) {
      queue.shift();
      if (queue.length > 0) {
        audio.currentTime = queue[0].start;
      } else {
        audio.pause();
      }
    }
  };

  const playPair = (aId: string, bId: string) => {
    const audio = audioRef.current;
    const a = result?.segments.find((s) => s.id === aId);
    const b = result?.segments.find((s) => s.id === bId);
    if (!audio || !a || !b) return;
    playQueueRef.current =
      aId === bId ? [{ start: a.start, end: a.end }] : [a, b].map((s) => ({ start: s.start, end: s.end }));
    audio.currentTime = playQueueRef.current[0].start;
    audio.play().catch(() => undefined);
  };

  useEffect(() => {
    const loadRecordings = async () => {
      try {
        const response = await fetch(`${API_BASE}/tasks/task-b/dataset/recordings`, {
          credentials: "include",
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not list recordings.");
        setRecordings(payload.recordings as RecordingInfo[]);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not list recordings.");
      }
    };
    const loadUploads = async () => {
      try {
        const response = await fetch(`${API_BASE}/tasks/task-b/uploads`, {
          credentials: "include",
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not list your uploads.");
        setUploads(payload.uploads as RecordingInfo[]);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not list your uploads.");
      }
    };
    const loadModels = async () => {
      try {
        const response = await fetch(`${API_BASE}/tasks/task-b/models`, {
          credentials: "include",
        });
        const payload = await response.json();
        if (!response.ok) return;
        setModels(payload.models as DiarizationModelInfo[]);
      } catch {
        // The note is explanatory only -- losing it must not surface an error
        // banner over a workbench that is otherwise working.
      }
    };
    loadRecordings();
    // Uploads live for the session, so a refresh has to restore them.
    loadUploads();
    loadModels();
  }, []);

  const uploadAudio = async (file: File) => {
    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(`${API_BASE}/tasks/task-b/uploads`, {
        method: "POST",
        credentials: "include",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Upload failed (${response.status})`);
      }
      const uploaded = payload as RecordingInfo;
      // Newest first, matching the order the backend lists them in.
      setUploads((current) => [uploaded, ...current]);
      // Select it right away — uploading it is the intent to diarize it.
      setSelectedRecordingId(uploaded.recording_id);
      setResult(null);
      setProjection(null);
      setSelectedId(null);
      resetPerturbation();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const runDiarization = async () => {
    if (!selectedRecordingId || !model) return;
    setIsRunning(true);
    setError(null);
    setResult(null);
    setProjection(null);
    setSelectedId(null);

    try {
      const runResponse = await fetch(`${API_BASE}/tasks/task-b/run`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, recording_id: selectedRecordingId }),
      });
      const runPayload = await runResponse.json();
      if (!runResponse.ok) {
        throw new Error(runPayload.detail || `Diarization failed (${runResponse.status})`);
      }
      setResult(runPayload as DiarizationResult);

      const projectionResponse = await fetch(
        `${API_BASE}/tasks/task-b/projection?model=${encodeURIComponent(
          model
        )}&recording_id=${encodeURIComponent(selectedRecordingId)}`,
        { credentials: "include" }
      );
      const projectionPayload = await projectionResponse.json();
      if (projectionResponse.ok) {
        setProjection(projectionPayload as ProjectionResult);
      }
      // A projection error (e.g. too few segments) should not hide the timeline.
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Diarization failed.");
    } finally {
      setIsRunning(false);
    }
  };

  /** A model switch invalidates everything on screen: the timeline, the
   *  projection and the perturbation delta all belong to the model that
   *  produced them, and leaving them up would label one model's result with
   *  another's name. Clear them, then re-run — `/run` is cache-through, so a
   *  model already computed for this recording comes back immediately, and an
   *  uncached one costs the same whether a button or this effect starts it.
   *  Uploads and the recording list are deliberately untouched: which audio is
   *  selected does not change when the model does. */
  const isFirstModelRender = useRef(true);
  useEffect(() => {
    if (isFirstModelRender.current) {
      isFirstModelRender.current = false;
      return;
    }
    setResult(null);
    setProjection(null);
    setSelectedId(null);
    resetPerturbation();
    if (selectedRecordingId) runDiarization();
    // Only a model change triggers this; runDiarization reads the latest
    // selection from the render it was created in.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  /** Empty until /models resolves, and for a key the backend does not know —
   *  a missing note renders nothing rather than breaking the header. */
  const glassBoxNote = models.find((m) => m.key === model)?.glass_box_note ?? "";

  const runPerturbation = async () => {
    if (!selectedRecordingId || !model) return;

    perturbationAbortRef.current?.abort();
    const controller = new AbortController();
    perturbationAbortRef.current = controller;

    setIsPerturbing(true);
    setPerturbationError(null);
    setPerturbation(null);

    const params =
      perturbationType === "noise"
        ? { noise_level: noisePercentToLevel(noisePercent) }
        : { mask_start_percent: maskRange[0], mask_end_percent: maskRange[1] };

    try {
      const response = await fetch(`${API_BASE}/tasks/task-b/perturbation`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model,
          recording_id: selectedRecordingId,
          perturbation: { type: perturbationType, params },
        }),
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || `Perturbation failed (${response.status})`);
      }
      setPerturbation(payload as PerturbationResult);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      setPerturbationError(caught instanceof Error ? caught.message : "Perturbation failed.");
    } finally {
      setIsPerturbing(false);
      if (perturbationAbortRef.current === controller) {
        perturbationAbortRef.current = null;
      }
    }
  };

  /** Seek the perturbed clip rather than the original — the two runs have
   *  their own audio, and clicking a perturbed segment should play what that
   *  run actually heard. */
  const seekPerturbedSegment = (segmentId: string) => {
    setSelectedId(segmentId);
    const segment = perturbation?.perturbed.segments.find((s) => s.id === segmentId);
    const audio = perturbedAudioRef.current;
    if (segment && audio) {
      audio.currentTime = segment.start;
      audio.play().catch(() => undefined);
    }
  };

  const seekToSegment = (segmentId: string) => {
    setSelectedId(segmentId);
    const segment = result?.segments.find((s) => s.id === segmentId);
    const audio = audioRef.current;
    if (segment && audio) {
      audio.currentTime = segment.start;
      audio.play().catch(() => undefined);
    }
  };

  const audioUrl = selectedRecordingId ? audioUrlFor(selectedRecordingId) : undefined;

  return (
    <div className="h-full overflow-y-auto bg-background p-4 scrollbar-thin">
      <div className="mx-auto max-w-4xl space-y-4">
        <div>
          <div className="flex items-center gap-2">
            <AudioLines className="h-5 w-5 text-primary" />
            <h2 className="text-lg font-semibold">Glass-Box Speaker Diarization</h2>
            <Badge variant="outline">{modelLabel}</Badge>
          </div>
          <p className="mt-1 text-sm text-muted-foreground">
            Who spoke when — with the pipeline's own internals exposed: per-segment
            confidence from the clustering's embedding space, hatched where uncertain.
          </p>
          {glassBoxNote && (
            <p className="mt-1 text-[10px] text-muted-foreground">{glassBoxNote}</p>
          )}
        </div>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        <Card>
          <CardHeader>
            <CardTitle className="text-sm">1. Pick a recording</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex flex-wrap items-center gap-2">
              <select
                className="h-9 rounded-md border bg-background px-2 text-sm"
                value={selectedRecordingId}
                onChange={(event) => {
                  setSelectedRecordingId(event.target.value);
                  setResult(null);
                  setProjection(null);
                  setSelectedId(null);
                  setError(null);
                  resetPerturbation();
                }}
              >
                <option value="">Select a recording…</option>
                <optgroup label="AMI meetings">
                  {recordings.map((recording) => (
                    <option key={recording.recording_id} value={recording.recording_id}>
                      {recording.display_filename}
                    </option>
                  ))}
                </optgroup>
                {uploads.length > 0 && (
                  <optgroup label="Your uploads">
                    {uploads.map((upload) => (
                      <option key={upload.recording_id} value={upload.recording_id}>
                        {upload.display_filename}
                      </option>
                    ))}
                  </optgroup>
                )}
              </select>
              <Button
                variant="outline"
                onClick={() => uploadInputRef.current?.click()}
                disabled={isUploading || isRunning}
              >
                {isUploading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Uploading…
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" /> Upload audio
                  </>
                )}
              </Button>
              <input
                ref={uploadInputRef}
                type="file"
                accept="audio/wav,audio/mpeg,audio/mp4,audio/flac,.wav,.mp3,.m4a,.flac"
                className="hidden"
                onChange={(event) => {
                  const file = event.target.files?.[0];
                  if (file) uploadAudio(file);
                  // Reset so re-picking the same file fires onChange again.
                  event.target.value = "";
                }}
              />
              <Button onClick={runDiarization} disabled={!selectedRecordingId || isRunning}>
                {isRunning ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Diarizing… (first run
                    on a full meeting takes minutes)
                  </>
                ) : (
                  <>
                    <Play className="mr-2 h-4 w-4" /> Run diarization
                  </>
                )}
              </Button>
            </div>
            <p className="text-xs text-muted-foreground">
              Upload your own WAV, MP3, M4A or FLAC (max 50 MB). Uploads are private to
              your session; a first run costs minutes of CPU, but re-running the same
              audio is instant.
            </p>
            {audioUrl && (
              <audio
                ref={audioRef}
                controls
                className="w-full"
                src={audioUrl}
                onTimeUpdate={handleTimeUpdate}
              />
            )}
          </CardContent>
        </Card>

        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                2. Speaker timeline{" "}
                <span className="font-normal text-muted-foreground">
                  — {result.num_speakers} speakers, {result.segments.length} segments
                  {result.cached ? " (cached)" : ""}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <DiarizationTimeline
                segments={result.segments}
                speakers={result.speakers}
                duration={result.duration}
                hoveredId={hoveredId}
                selectedId={selectedId}
                onHover={setHoveredId}
                onSelect={seekToSegment}
              />
            </CardContent>
          </Card>
        )}

        {result && projection && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                3. Segment embeddings (PCA){" "}
                <span className="font-normal text-muted-foreground">
                  — each dot is one segment in the space the clustering used
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <EmbeddingScatter
                points={projection.points}
                speakers={result.speakers}
                hoveredId={hoveredId}
                selectedId={selectedId}
                onHover={setHoveredId}
                onSelect={seekToSegment}
              />
            </CardContent>
          </Card>
        )}

        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                4. Segment similarity matrix{" "}
                <span className="font-normal text-muted-foreground">
                  — example-based: click any cell to hear both segments
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <SimilarityMatrix
                segments={result.segments}
                embeddings={result.embeddings}
                speakers={result.speakers}
                onPlayPair={playPair}
              />
            </CardContent>
          </Card>
        )}

        {result && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                5. Perturbation counterfactual{" "}
                <span className="font-normal text-muted-foreground">
                  — degrade the audio, re-diarize, and see what the pipeline
                  changes its mind about
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <PerturbationControls
                activeType={perturbationType}
                onActiveTypeChange={(type) => {
                  setPerturbationType(type);
                  resetPerturbation();
                }}
                noisePercent={noisePercent}
                onNoisePercentChange={(value) => {
                  setNoisePercent(value);
                  resetPerturbation();
                }}
                maskRange={maskRange}
                onMaskRangeChange={(value) => {
                  setMaskRange(value);
                  resetPerturbation();
                }}
                onRun={runPerturbation}
                isRunning={isPerturbing}
                disabled={!selectedRecordingId || isRunning}
                disabledReason={
                  isRunning ? "Waiting for the current diarization to finish." : null
                }
              />

              {perturbationError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Perturbation could not be completed</AlertTitle>
                  <AlertDescription>{perturbationError}</AlertDescription>
                </Alert>
              )}

              {perturbation && (
                <div className="space-y-4">
                  <DeltaSummaryCard
                    delta={perturbation.delta}
                    perturbation={perturbation.perturbation}
                    cached={perturbation.cached}
                  />

                  <StackedTimelines
                    original={perturbation.original}
                    perturbed={perturbation.perturbed}
                    delta={perturbation.delta}
                    hoveredId={hoveredId}
                    selectedId={selectedId}
                    onHover={setHoveredId}
                    onSelectOriginal={seekToSegment}
                    onSelectPerturbed={seekPerturbedSegment}
                  />

                  <div className="space-y-1">
                    <div className="text-xs text-muted-foreground">
                      Perturbed audio — listen to what the second run actually heard
                    </div>
                    <audio
                      ref={perturbedAudioRef}
                      controls
                      className="w-full"
                      src={`${API_BASE}/tasks/task-b/perturbed/${encodeURIComponent(
                        perturbation.perturbed.recording_id
                      )}/audio`}
                    />
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        )}

      </div>
    </div>
  );
};
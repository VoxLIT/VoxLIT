import { useEffect, useRef, useState } from "react";
import { API_BASE } from "@/lib/api";
import { noisePercentToLevel } from "../PerturbationControls";
import {
  DiarizationModelInfo,
  DiarizationResult,
  PerturbationResult,
  PerturbationType,
  ProjectionResult,
  RecordingInfo,
} from "../types";

/** Session uploads and demo recordings differ only by id prefix; the audio
 *  endpoint differs, everything downstream does not. */
const UPLOAD_ID_PREFIX = "upl_";

export const audioUrlFor = (recordingId: string): string =>
  recordingId.startsWith(UPLOAD_ID_PREFIX)
    ? `${API_BASE}/tasks/task-b/uploads/${encodeURIComponent(recordingId)}/audio`
    : `${API_BASE}/tasks/task-b/dataset/recordings/${encodeURIComponent(recordingId)}/audio`;

/**
 * All of the diarization workbench's state and fetch logic: recordings and
 * uploads, the run + projection, the hover/selection shared by the timeline,
 * scatter and matrix, the perturbation counterfactual, and audio playback.
 * Views render from what this returns and own no state of their own.
 */
export function useDiarization(model: string) {
  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [uploads, setUploads] = useState<RecordingInfo[]>([]);
  const [models, setModels] = useState<DiarizationModelInfo[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [selectedRecordingId, setSelectedRecordingId] = useState<string>("");
  const [result, setResult] = useState<DiarizationResult | null>(null);
  const [projection, setProjection] = useState<ProjectionResult | null>(null);
  // Kept apart from `error`: a map that cannot be drawn is not a failed run.
  const [projectionError, setProjectionError] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const audioRef = useRef<HTMLAudioElement>(null);

  // --- Perturbation counterfactual -------------------------------------
  const [perturbationType, setPerturbationTypeState] = useState<PerturbationType>("noise");
  const [noisePercent, setNoisePercentState] = useState(4);
  const [maskRange, setMaskRangeState] = useState<[number, number]>([20, 40]);
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

  // Changing any perturbation setting invalidates the current comparison.
  const setPerturbationType = (type: PerturbationType) => {
    setPerturbationTypeState(type);
    resetPerturbation();
  };
  const setNoisePercent = (value: number) => {
    setNoisePercentState(value);
    resetPerturbation();
  };
  const setMaskRange = (value: [number, number]) => {
    setMaskRangeState(value);
    resetPerturbation();
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

  /** Pick which audio is under test; everything computed for the previous
   *  one no longer applies. */
  const select = (recordingId: string) => {
    setSelectedRecordingId(recordingId);
    setResult(null);
    setProjection(null);
    setProjectionError(null);
    setSelectedId(null);
    setError(null);
    resetPerturbation();
  };

  const upload = async (file: File) => {
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
      setProjectionError(null);
      setSelectedId(null);
      resetPerturbation();
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  };

  const run = async () => {
    if (!selectedRecordingId || !model) return;
    setIsRunning(true);
    setError(null);
    setResult(null);
    setProjection(null);
    setProjectionError(null);
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
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Diarization failed.");
      setIsRunning(false);
      return;
    }

    // Its own try: a projection failure (e.g. too few segments, 422) must not
    // hide the timeline or show up as a failed run.
    try {
      const projectionResponse = await fetch(
        `${API_BASE}/tasks/task-b/projection?model=${encodeURIComponent(
          model
        )}&recording_id=${encodeURIComponent(selectedRecordingId)}`,
        { credentials: "include" }
      );
      const projectionPayload = await projectionResponse.json().catch(() => ({}));
      if (!projectionResponse.ok) {
        throw new Error(projectionPayload.detail || `Projection failed (${projectionResponse.status})`);
      }
      setProjection(projectionPayload as ProjectionResult);
    } catch (caught) {
      setProjectionError(caught instanceof Error ? caught.message : "Projection failed.");
    } finally {
      setIsRunning(false);
    }
  };

  /** A model switch invalidates everything on screen: the timeline, the
   *  projection and the perturbation delta all belong to the model that
   *  produced them, and leaving them up would label one model's result with
   *  another's name. Clear them and wait -- the user re-runs explicitly, since
   *  an uncached run under a new model costs minutes of CPU. Uploads, the
   *  recording list and the selected recording are deliberately untouched. */
  useEffect(() => {
    setResult(null);
    setProjection(null);
    setProjectionError(null);
    setSelectedId(null);
    resetPerturbation();
    // Only a model change triggers this; resetPerturbation touches refs and
    // setters only, so a stale copy of it is harmless.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [model]);

  /** Empty until /models resolves, and for a key the backend does not know —
   *  a missing note renders nothing rather than breaking the header. */
  const glassBoxNote = models.find((m) => m.key === model)?.glass_box_note ?? "";
  const embeddingDimension = models.find((m) => m.key === model)?.embedding_dimension ?? null;

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
  const perturbedAudioUrl = perturbation
    ? `${API_BASE}/tasks/task-b/perturbed/${encodeURIComponent(perturbation.perturbed.recording_id)}/audio`
    : undefined;

  return {
    // recordings + uploads
    recordings,
    uploads,
    isUploading,
    upload,
    selectedRecordingId,
    select,
    // run
    result,
    projection,
    projectionError,
    isRunning,
    error,
    run,
    glassBoxNote,
    embeddingDimension,
    // shared hover / selection
    hoveredId,
    setHoveredId,
    selectedId,
    // perturbation
    perturbationType,
    setPerturbationType,
    noisePercent,
    setNoisePercent,
    maskRange,
    setMaskRange,
    perturbation,
    isPerturbing,
    perturbationError,
    runPerturbation,
    // audio
    audioRef,
    perturbedAudioRef,
    audioUrl,
    perturbedAudioUrl,
    handleTimeUpdate,
    seekToSegment,
    seekPerturbedSegment,
    playPair,
  };
}

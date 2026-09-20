import { useEffect, useState } from "react";
import { AlertCircle, Box, HelpCircle, Play, RefreshCw, Square } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { EmbeddingScatter } from "./EmbeddingScatter";
import { BONAFIDE_COLOR, SPOOF_COLOR } from "./embeddingColors";
import type { DeepfakeEmbeddingProjection } from "./types";
import type { UploadedFile } from "@/tasks/types";

type ReductionMethod = "pca" | "umap" | "tsne";

const METHOD_LABELS: Record<ReductionMethod, string> = {
  pca: "PCA",
  umap: "UMAP",
  tsne: "t-SNE",
};

interface DeepfakeEmbeddingPanelProps {
  model: string;
  modelLabel: string;
  /** Recordings listed for the dataset; the view stays idle until there are some. */
  availableFiles: string[];
  /** Shared selection, so a point, a table row and the workbench picker agree. */
  selectedFile: UploadedFile | null;
  onFileSelect: (file: UploadedFile) => void;
}

/**
 * The deepfake task's own version of the left "Audio Embeddings" panel: the
 * whole dataset in 2D or 3D.
 *
 * Each recording is one point — the vector the detector's classification head
 * reads, reduced with PCA, UMAP or t-SNE — coloured by the detector's own spoof
 * score, never by the dataset's labels (see the backend's embeddings.py), so
 * the view shows how the detector groups the clips without giving away the
 * answer the workbench asks the user to judge.
 *
 * Built for this task rather than reusing the shared EmbeddingPanel, whose data
 * path is the legacy /inferences/embeddings endpoint (Whisper / Wav2Vec2 only).
 *
 * Nothing runs until the button is pressed: a model's first projection scores
 * the whole dataset (and downloads the checkpoint). After that, changing the
 * method or 2D/3D re-projects automatically and is quick.
 */
export const DeepfakeEmbeddingPanel = ({
  model,
  modelLabel,
  availableFiles,
  selectedFile,
  onFileSelect,
}: DeepfakeEmbeddingPanelProps) => {
  const [started, setStarted] = useState(false);
  const [method, setMethod] = useState<ReductionMethod>("pca");
  const [is3D, setIs3D] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);
  const [projection, setProjection] = useState<DeepfakeEmbeddingProjection | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const datasetAvailable = availableFiles.length > 0;

  useEffect(() => {
    if (!started || !datasetAvailable) return;

    const controller = new AbortController();
    setIsLoading(true);
    setError(null);

    const load = async () => {
      try {
        const response = await fetch(`${API_BASE}/tasks/deepfake/embeddings`, {
          method: "POST",
          credentials: "include",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model, reduction_method: method, n_components: is3D ? 3 : 2 }),
          signal: controller.signal,
        });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail || `Embedding view failed (${response.status})`);
        }
        setProjection(payload as DeepfakeEmbeddingProjection);
        setIsLoading(false);
      } catch (caught) {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setError(caught instanceof Error ? caught.message : "Embedding view failed.");
        setIsLoading(false);
      }
    };
    load();

    return () => controller.abort();
  }, [started, datasetAvailable, model, method, is3D, refreshToken]);

  // Points from another detector would silently belong to the wrong model.
  const visible = projection !== null && projection.model === model ? projection : null;

  const selectRecording = (recordingId: string) => {
    const recording = visible?.recordings.find((candidate) => candidate.recording_id === recordingId);
    if (!recording) return;
    onFileSelect({
      file_id: recording.recording_id,
      filename: recording.display_filename,
      file_path: recording.display_filename,
      message: "Selected from embeddings",
    });
  };

  const selectedRecording = visible?.recordings.find(
    (recording) => recording.recording_id === selectedFile?.file_id,
  );

  return (
    <TooltipProvider>
      <div className="h-full bg-white border-r border-gray-200 flex flex-col">
        <div className="panel-header p-3 border-b border-gray-200">
          <h3 className="font-bold text-sm text-gray-800 flex items-center gap-1.5">
            Audio Embeddings
            <Tooltip>
              <TooltipTrigger>
                <HelpCircle className="h-3.5 w-3.5 text-muted-foreground" />
              </TooltipTrigger>
              <TooltipContent className="font-normal">
                The dataset in 2D/3D: one point per recording, from the vector the detector reads
              </TooltipContent>
            </Tooltip>
          </h3>
        </div>

        <div className="flex-1 min-h-0 p-3 bg-panel-background overflow-y-auto scrollbar-thin">
          <div className="space-y-3">
            <div className="space-y-2.5">
              <div className="flex items-center justify-between">
                <Badge variant="outline" className="text-[10px] bg-primary/10 text-primary border-primary/20">
                  {modelLabel}
                </Badge>
                <div className="flex items-center gap-1.5">
                  <Select value={method} onValueChange={(value) => setMethod(value as ReductionMethod)}>
                    <SelectTrigger className="w-20 h-7 text-xs" aria-label="Dimensionality reduction method">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {(Object.keys(METHOD_LABELS) as ReductionMethod[]).map((key) => (
                        <SelectItem key={key} value={key}>
                          {METHOD_LABELS[key]}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Tooltip>
                    <TooltipTrigger asChild>
                      <Button
                        size="sm"
                        variant="secondary"
                        className="h-7 w-7 p-0"
                        disabled={!started || isLoading}
                        onClick={() => setRefreshToken((token) => token + 1)}
                        aria-label="Refresh embeddings visualization"
                      >
                        <RefreshCw className={`h-3 w-3 ${isLoading ? "animate-spin text-primary" : ""}`} />
                      </Button>
                    </TooltipTrigger>
                    <TooltipContent>Refresh embeddings visualization</TooltipContent>
                  </Tooltip>
                </div>
              </div>

              <div className="flex items-center gap-1.5 h-8 px-2 bg-gray-50 rounded-md border border-border w-fit">
                <Switch id="deepfake-3d-mode" checked={is3D} onCheckedChange={setIs3D} />
                <Label htmlFor="deepfake-3d-mode" className="text-[11px] flex items-center gap-1 font-medium cursor-pointer">
                  {is3D ? <Box className="h-3 w-3 text-primary" /> : <Square className="h-3 w-3 text-muted-foreground" />}
                  <span className={is3D ? "text-primary" : "text-muted-foreground"}>{is3D ? "3D" : "2D"}</span>
                </Label>
              </div>

              {!started && (
                <div className="space-y-2 p-3 bg-primary/5 rounded-sm border border-primary/20">
                  <p className="text-xs text-muted-foreground">
                    Plot all {datasetAvailable ? availableFiles.length : ""} recordings as points, coloured by
                    this detector&apos;s own score — not by the dataset&apos;s labels, which stay hidden.
                  </p>
                  <Button size="sm" className="w-full" onClick={() => setStarted(true)} disabled={!datasetAvailable}>
                    <Play className="mr-2 h-3.5 w-3.5" />
                    {datasetAvailable ? "Show the dataset in 2D / 3D" : "Waiting for the recording list…"}
                  </Button>
                </div>
              )}

              {isLoading && (
                <div className="text-xs text-primary flex items-start gap-2 p-3 bg-primary/5 rounded-sm border border-primary/20">
                  <div className="mt-1 w-2 h-2 shrink-0 bg-primary rounded-full animate-ping"></div>
                  <span>
                    {visible
                      ? "Re-projecting…"
                      : "Scoring every recording with this detector. The first run for a model takes minutes (and downloads the checkpoint); after that this view is quick."}
                  </span>
                </div>
              )}

              {error && (
                <div className="text-xs text-destructive flex items-start gap-2 p-3 bg-destructive/5 rounded-sm border border-destructive/20">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>{error}</span>
                </div>
              )}

              {visible && visible.reduction_method_used !== visible.reduction_method && (
                <div className="flex items-start gap-2 rounded-md bg-amber-50 p-2 text-[11px] text-amber-800">
                  <AlertCircle className="mt-0.5 h-3.5 w-3.5 shrink-0" />
                  <span>
                    {visible.reduction_method.toUpperCase()} could not run on this data — showing{" "}
                    {visible.reduction_method_used.toUpperCase()} instead.
                  </span>
                </div>
              )}
            </div>

            {visible && (
              <>
                <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
                  <span>bona fide</span>
                  <div
                    className="h-2 flex-1 rounded-full"
                    style={{ background: `linear-gradient(to right, ${BONAFIDE_COLOR}, ${SPOOF_COLOR})` }}
                    aria-hidden="true"
                  />
                  <span>spoof</span>
                </div>

                <div className="h-[450px] border border-border rounded-lg bg-card p-1.5 overflow-hidden">
                  <EmbeddingScatter
                    recordings={visible.recordings}
                    coordinates={visible.coordinates}
                    is3D={visible.n_components === 3}
                    viewKey={`${visible.reduction_method_used}-${visible.n_components}`}
                    selectedRecordingId={selectedFile?.file_id ?? ""}
                    onSelectRecording={selectRecording}
                  />
                </div>

                {selectedRecording && (
                  <p className="text-[11px] font-mono text-muted-foreground">
                    {selectedRecording.display_filename} · spoof score{" "}
                    {selectedRecording.spoof_probability.toFixed(3)}
                  </p>
                )}
                <p className="text-[11px] text-muted-foreground">
                  {visible.total_recordings} recordings · {visible.embedding_dimension}-dimensional vectors
                  reduced to {visible.n_components}D with{" "}
                  {METHOD_LABELS[visible.reduction_method_used as ReductionMethod] ?? visible.reduction_method_used}.
                  Click a point to select that recording. The projection is visual only: distances on screen
                  are not distances inside the model, least of all with t-SNE and UMAP.
                </p>
              </>
            )}
          </div>
        </div>
      </div>
    </TooltipProvider>
  );
};

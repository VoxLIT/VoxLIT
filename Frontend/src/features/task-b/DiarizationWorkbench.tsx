import { useRef } from "react";
import { AudioLines, Loader2, Play, Upload } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { DiarizationTimeline } from "./DiarizationTimeline";
import { SimilarityMatrix } from "./SimilarityMatrix";
import { EmbeddingScatter } from "./EmbeddingScatter";
import { DeltaSummaryCard } from "./DeltaSummaryCard";
import { PerturbationControls } from "./PerturbationControls";
import { StackedTimelines } from "./StackedTimelines";
import { useDiarization } from "./page/useDiarization";

interface DiarizationWorkbenchProps {
  model: string;
  modelLabel: string;
}

export const DiarizationWorkbench = ({ model, modelLabel }: DiarizationWorkbenchProps) => {
  const d = useDiarization(model);
  const uploadInputRef = useRef<HTMLInputElement>(null);

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
          {d.glassBoxNote && (
            <p className="mt-1 text-[10px] text-muted-foreground">{d.glassBoxNote}</p>
          )}
        </div>

        {d.error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Something went wrong</AlertTitle>
            <AlertDescription>{d.error}</AlertDescription>
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
                value={d.selectedRecordingId}
                onChange={(event) => d.select(event.target.value)}
              >
                <option value="">Select a recording…</option>
                <optgroup label="AMI meetings">
                  {d.recordings.map((recording) => (
                    <option key={recording.recording_id} value={recording.recording_id}>
                      {recording.display_filename}
                    </option>
                  ))}
                </optgroup>
                {d.uploads.length > 0 && (
                  <optgroup label="Your uploads">
                    {d.uploads.map((upload) => (
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
                disabled={d.isUploading || d.isRunning}
              >
                {d.isUploading ? (
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
                  if (file) d.upload(file);
                  // Reset so re-picking the same file fires onChange again.
                  event.target.value = "";
                }}
              />
              <Button onClick={d.run} disabled={!d.selectedRecordingId || d.isRunning}>
                {d.isRunning ? (
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
            {d.audioUrl && (
              <audio
                ref={d.audioRef}
                controls
                className="w-full"
                src={d.audioUrl}
                onTimeUpdate={d.handleTimeUpdate}
              />
            )}
          </CardContent>
        </Card>

        {d.result && (
          <Card>
            <CardHeader>
              <CardTitle className="text-sm">
                2. Speaker timeline{" "}
                <span className="font-normal text-muted-foreground">
                  — {d.result.num_speakers} speakers, {d.result.segments.length} segments
                  {d.result.cached ? " (cached)" : ""}
                </span>
              </CardTitle>
            </CardHeader>
            <CardContent>
              <DiarizationTimeline
                segments={d.result.segments}
                speakers={d.result.speakers}
                duration={d.result.duration}
                hoveredId={d.hoveredId}
                selectedId={d.selectedId}
                onHover={d.setHoveredId}
                onSelect={d.seekToSegment}
              />
            </CardContent>
          </Card>
        )}

        {d.result && d.projection && (
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
                points={d.projection.points}
                speakers={d.result.speakers}
                hoveredId={d.hoveredId}
                selectedId={d.selectedId}
                onHover={d.setHoveredId}
                onSelect={d.seekToSegment}
              />
            </CardContent>
          </Card>
        )}

        {d.result && (
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
                segments={d.result.segments}
                embeddings={d.result.embeddings}
                speakers={d.result.speakers}
                onPlayPair={d.playPair}
              />
            </CardContent>
          </Card>
        )}

        {d.result && (
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
                activeType={d.perturbationType}
                onActiveTypeChange={d.setPerturbationType}
                noisePercent={d.noisePercent}
                onNoisePercentChange={d.setNoisePercent}
                maskRange={d.maskRange}
                onMaskRangeChange={d.setMaskRange}
                onRun={d.runPerturbation}
                isRunning={d.isPerturbing}
                disabled={!d.selectedRecordingId || d.isRunning}
                disabledReason={
                  d.isRunning ? "Waiting for the current diarization to finish." : null
                }
              />

              {d.perturbationError && (
                <Alert variant="destructive">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Perturbation could not be completed</AlertTitle>
                  <AlertDescription>{d.perturbationError}</AlertDescription>
                </Alert>
              )}

              {d.perturbation && (
                <div className="space-y-4">
                  <DeltaSummaryCard
                    delta={d.perturbation.delta}
                    perturbation={d.perturbation.perturbation}
                    cached={d.perturbation.cached}
                  />

                  <StackedTimelines
                    original={d.perturbation.original}
                    perturbed={d.perturbation.perturbed}
                    delta={d.perturbation.delta}
                    hoveredId={d.hoveredId}
                    selectedId={d.selectedId}
                    onHover={d.setHoveredId}
                    onSelectOriginal={d.seekToSegment}
                    onSelectPerturbed={d.seekPerturbedSegment}
                  />

                  <div className="space-y-1">
                    <div className="text-xs text-muted-foreground">
                      Perturbed audio — listen to what the second run actually heard
                    </div>
                    <audio
                      ref={d.perturbedAudioRef}
                      controls
                      className="w-full"
                      src={d.perturbedAudioUrl}
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
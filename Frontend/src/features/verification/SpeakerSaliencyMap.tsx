import { useMemo, useRef, useState } from "react";
import type WaveSurfer from "wavesurfer.js";
import { AlertCircle } from "lucide-react";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Slider } from "@/components/ui/slider";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { WaveformViewer } from "@/components/audio/WaveformViewer";
import { InfoTooltip } from "./InfoTooltip";
import {
  isFrequencySaliency,
  type AnySaliencyMapResponse,
  type SaliencyAxis,
  type SaliencySegment,
} from "./saliencyTypes";

// A deliberately conservative *visualization* cutoff, not a claim about
// measured embedding-extraction noise -- no such calibration exists yet, so
// this does not assert one. Below this per-segment cosine-similarity-change
// magnitude, the map is treated as too small to usefully distinguish "high"
// from "low" influence on a per-result normalized scale.
const MIN_DISPLAY_INFLUENCE_DELTA = 0.01;

const SEGMENT_COUNT_MIN = 4;
const SEGMENT_COUNT_MAX = 20;

function intensityToColor(v: number): string {
  const clamp = (x: number) => Math.max(0, Math.min(1, x));
  const mix = (a: number, b: number, t: number) => a + (b - a) * t;
  // low: light blue -> mid: yellow -> high: orange/red
  let h: number, s: number, l: number;
  if (v < 0.5) {
    const t = clamp(v / 0.5);
    h = mix(205, 45, t);
    s = mix(85, 93, t);
    l = mix(88, 58, t);
  } else {
    const t = clamp((v - 0.5) / 0.5);
    h = mix(45, 9, t);
    s = mix(93, 86, t);
    l = mix(58, 50, t);
  }
  return `hsl(${h} ${s}% ${l}%)`;
}

type SegmentClassification = "supports" | "opposes" | "minimal";

function classifySegment(
  segment: Pick<SaliencySegment, "influence_strength" | "similarity_change">
): SegmentClassification {
  if (segment.influence_strength < MIN_DISPLAY_INFLUENCE_DELTA) return "minimal";
  return segment.similarity_change > 0 ? "supports" : "opposes";
}

const CLASSIFICATION_LABEL: Record<SegmentClassification, string> = {
  supports: "Supports similarity",
  opposes: "Opposes similarity",
  minimal: "Minimal influence",
};

const CLASSIFICATION_UNDERLINE: Record<SegmentClassification, string> = {
  supports: "#10b981", // emerald-500
  opposes: "#f43f5e", // rose-500
  minimal: "#9ca3af", // gray-400
};

const formatScore = (value: number) => value.toFixed(4);
const formatSigned = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
const formatTime = (seconds: number) => `${seconds.toFixed(2)}s`;
const formatHzRange = (lowHz: number, highHz: number) => `${Math.round(lowHz)}–${Math.round(highHz)} Hz`;

// Band changes are shown compactly; toFixed keeps the minus sign, so only "+" is added.
const formatSignedShort = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;

/** Explains the baseline score that every occlusion bar is measured against. */
const BaselineSimilarityCard = ({ result }: { result: AnySaliencyMapResponse }) => {
  const isCluster = result.reference_type === "cluster";
  const isAboveThreshold = result.baseline_similarity >= result.threshold;
  return (
    <div className="rounded-md border p-2.5" data-testid="baseline-similarity-card">
      <div className="flex items-center gap-1 text-xs text-muted-foreground">
        Baseline similarity
        <InfoTooltip title="Baseline similarity">
          <p>
            The model turns each reference clip into a voice fingerprint and averages them into one
            &apos;centroid&apos;. This number is the cosine similarity between this clip and that centroid.
          </p>
          <p>
            It is usually higher than &apos;Avg. similarity to cluster&apos;, which averages one-by-one comparisons,
            because averaging cancels out each clip&apos;s noise.
          </p>
          <p>Each bar below shows how much this number drops when that part is silenced.</p>
        </InfoTooltip>
      </div>
      <div className="text-2xl font-semibold tabular-nums leading-tight">
        {formatScore(result.baseline_similarity)}
      </div>
      {isCluster ? (
        // No threshold comparison here: result.threshold is the
        // pair-verification threshold, not the clustering threshold.
        <p className="text-xs text-muted-foreground">
          {result.reference_count === 1
            ? "How similar this clip is to the other clip in this speaker group (this clip excluded)."
            : `How similar this clip is to the average voice of the other ${result.reference_count} clips in this speaker group (this clip excluded).`}
        </p>
      ) : (
        <>
          <p className="text-xs text-muted-foreground">
            How similar this clip is to the reference speaker&apos;s average voice, before anything is silenced.
          </p>
          <p className={`text-xs font-medium ${isAboveThreshold ? "text-emerald-700" : "text-rose-700"}`}>
            {isAboveThreshold
              ? `Above threshold ${result.threshold.toFixed(2)} → same speaker`
              : `Below threshold ${result.threshold.toFixed(2)} → different speakers`}
          </p>
        </>
      )}
    </div>
  );
};

export interface SpeakerSaliencyMapProps {
  title: string;
  audioUrl: string | undefined;
  requireCredentials: boolean;
  /** The result for the currently selected occlusion axis. */
  result: AnySaliencyMapResponse | null;
  isLoading: boolean;
  error: string | null;
  /** Non-null: the displayed result no longer matches the current
   *  model/selection/segment-count -- dim it and explain why, rather than
   *  silently presenting it as current or removing it outright. */
  staleReason: string | null;
  /** Non-null: no Generate action is offered at all (e.g. singleton
   *  cluster, no valid batch/verification result) -- this replaces the
   *  Generate button and any prior result entirely. */
  emptyStateMessage: string | null;
  onGenerate: (() => void) | null;
  generateLabel: string;
  segmentCount: number;
  onSegmentCountChange: (segmentCount: number) => void;
  clusterBadge?: { label: string; color: string } | null;
  /** Which occlusion view is shown. Uncontrolled (starting at "time") when
   *  omitted. */
  occlusionAxis?: SaliencyAxis;
  onOcclusionAxisChange?: (axis: SaliencyAxis) => void;
}

export const SpeakerSaliencyMap = ({
  title,
  audioUrl,
  requireCredentials,
  result: axisResult,
  isLoading,
  error,
  staleReason,
  emptyStateMessage,
  onGenerate,
  generateLabel,
  segmentCount,
  onSegmentCountChange,
  clusterBadge,
  occlusionAxis,
  onOcclusionAxisChange,
}: SpeakerSaliencyMapProps) => {
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const [internalAxis, setInternalAxis] = useState<SaliencyAxis>("time");
  const axis = occlusionAxis ?? internalAxis;

  const handleAxisChange = (value: string) => {
    // Radix emits "" when the active item is clicked again -- keep the axis.
    if (value !== "time" && value !== "frequency") return;
    setInternalAxis(value);
    onOcclusionAxisChange?.(value);
  };

  // Each view only ever renders a result produced for its own axis.
  const result = axis === "time" && axisResult && !isFrequencySaliency(axisResult) ? axisResult : null;
  const frequencyResult = axis === "frequency" && isFrequencySaliency(axisResult) ? axisResult : null;

  const maxBandInfluence = frequencyResult
    ? Math.max(...frequencyResult.bands.map((band) => band.influence_strength), 0)
    : 0;
  const isBandResultBelowDisplayThreshold = maxBandInfluence < MIN_DISPLAY_INFLUENCE_DELTA;

  const maxInfluence = useMemo(
    () => (result ? Math.max(...result.segments.map((s) => s.influence_strength), 0) : 0),
    [result]
  );
  const isResultBelowDisplayThreshold = maxInfluence < MIN_DISPLAY_INFLUENCE_DELTA;

  const normalizedStrength = (influenceStrength: number): number => {
    if (isResultBelowDisplayThreshold) return 0; // paint everything at the low end
    return Math.min(1, influenceStrength / maxInfluence);
  };

  const seekTo = (segment: SaliencySegment) => {
    if (!result || !wavesurferRef.current) return;
    const fraction = result.audio_duration_seconds > 0 ? segment.start_seconds / result.audio_duration_seconds : 0;
    try {
      wavesurferRef.current.seekTo(Math.max(0, Math.min(1, fraction)));
    } catch {
      // Playback not ready yet -- seeking is a nice-to-have, never fatal.
    }
  };

  const heatmapStrip = result && (
    <div className="space-y-2">
      <div
        className="relative flex h-8 w-full overflow-hidden rounded border border-border"
        onMouseLeave={() => setHoveredIndex(null)}
      >
        {result.segments.map((segment, index) => {
          const classification = classifySegment(segment);
          const widthPct =
            result.audio_duration_seconds > 0
              ? ((segment.end_seconds - segment.start_seconds) / result.audio_duration_seconds) * 100
              : 100 / result.segments.length;
          return (
            <div
              key={segment.segment_index}
              className={`h-full cursor-pointer transition-opacity ${
                hoveredIndex !== null && hoveredIndex !== index ? "opacity-60" : ""
              }`}
              style={{
                width: `${widthPct}%`,
                backgroundColor: intensityToColor(normalizedStrength(segment.influence_strength)),
                borderBottom: `3px solid ${CLASSIFICATION_UNDERLINE[classification]}`,
              }}
              onMouseEnter={() => setHoveredIndex(index)}
              onClick={() => seekTo(segment)}
            />
          );
        })}
        {hoveredIndex !== null && (
          <div className="pointer-events-none absolute -top-2 left-2 z-10 -translate-y-full rounded border border-border bg-popover px-2 py-1 text-xs shadow">
            {(() => {
              const segment = result.segments[hoveredIndex];
              const classification = classifySegment(segment);
              return (
                <div className="space-y-0.5">
                  <div className="font-medium">
                    {formatTime(segment.start_seconds)}–{formatTime(segment.end_seconds)}
                  </div>
                  <div>Baseline similarity: {formatScore(result.baseline_similarity)}</div>
                  <div>Occluded similarity: {formatScore(segment.occluded_similarity)}</div>
                  <div>Change: {formatSigned(segment.similarity_change)}</div>
                  <div>Influence: {formatScore(segment.influence_strength)}</div>
                  <div className="font-medium">{CLASSIFICATION_LABEL[classification]}</div>
                </div>
              );
            })()}
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: intensityToColor(0) }} />
          Low influence
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: intensityToColor(0.5) }} />
          Medium influence
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: intensityToColor(1) }} />
          High influence
        </span>
      </div>
      {isResultBelowDisplayThreshold && (
        <p className="text-[10px] text-muted-foreground">
          All segment changes are below the minimum display-influence threshold.
        </p>
      )}
    </div>
  );

  const mostInfluential = result
    ? [...result.segments].sort((a, b) => b.influence_strength - a.influence_strength)
    : [];

  return (
    <Card>
      <CardHeader className="pb-2">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="flex items-center gap-2 text-sm">
            {title}
            {clusterBadge && (
              <Badge
                variant="outline"
                style={{ borderColor: clusterBadge.color, color: clusterBadge.color }}
                className="text-[10px]"
              >
                {clusterBadge.label}
              </Badge>
            )}
          </CardTitle>
        </div>
      </CardHeader>
      <CardContent className="space-y-3">
        {emptyStateMessage ? (
          <p className="text-xs text-muted-foreground">{emptyStateMessage}</p>
        ) : (
          <>
            <div className="flex flex-wrap items-center gap-3">
              <ToggleGroup
                type="single"
                size="sm"
                variant="outline"
                value={axis}
                onValueChange={handleAxisChange}
                aria-label="Occlusion axis"
              >
                <ToggleGroupItem value="time" className="h-7 px-2 text-xs">
                  Time
                </ToggleGroupItem>
                <ToggleGroupItem value="frequency" className="h-7 px-2 text-xs">
                  Frequency
                </ToggleGroupItem>
              </ToggleGroup>
              {axis === "time" ? (
                <div className="flex min-w-[10rem] flex-1 items-center gap-2 text-xs text-muted-foreground">
                  <span className="whitespace-nowrap">Segments: {segmentCount}</span>
                  <Slider
                    className="max-w-[10rem]"
                    min={SEGMENT_COUNT_MIN}
                    max={SEGMENT_COUNT_MAX}
                    step={1}
                    value={[segmentCount]}
                    onValueChange={([value]) => onSegmentCountChange(value)}
                    disabled={isLoading}
                    thumbLabel={`Segments: ${segmentCount}`}
                  />
                </div>
              ) : (
                <div className="flex-1" />
              )}
              <Button size="sm" variant="outline" onClick={() => onGenerate?.()} disabled={!onGenerate || isLoading}>
                {isLoading ? "Running occlusion passes…" : generateLabel}
              </Button>
            </div>

            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Saliency map could not be generated</AlertTitle>
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {staleReason && (
              <Alert className="border-amber-300 bg-amber-50">
                <AlertCircle className="h-4 w-4 text-amber-700" />
                <AlertDescription>{staleReason}</AlertDescription>
              </Alert>
            )}

            {result && (
              <div className={staleReason ? "space-y-3 opacity-50 pointer-events-none" : "space-y-3"}>
                <BaselineSimilarityCard result={result} />
                <WaveformViewer
                  audioUrl={audioUrl}
                  requireCredentials={requireCredentials}
                  onReady={(wavesurfer) => {
                    wavesurferRef.current = wavesurfer;
                  }}
                  timelineBelow={heatmapStrip}
                />

                <div className="space-y-1">
                  <div className="text-xs font-medium">Most influential segments</div>
                  {mostInfluential.map((segment) => {
                    const classification = classifySegment(segment);
                    return (
                      <div
                        key={segment.segment_index}
                        className={`flex items-center justify-between rounded px-2 py-1 text-xs cursor-pointer ${
                          hoveredIndex === segment.segment_index - 1 ? "bg-muted" : ""
                        }`}
                        onMouseEnter={() => setHoveredIndex(segment.segment_index - 1)}
                        onMouseLeave={() => setHoveredIndex(null)}
                        onClick={() => seekTo(segment)}
                      >
                        <span>
                          {formatTime(segment.start_seconds)}–{formatTime(segment.end_seconds)}
                        </span>
                        <span className="font-mono">{formatSigned(segment.similarity_change)}</span>
                        <span className="text-muted-foreground">{CLASSIFICATION_LABEL[classification]}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {axis === "frequency" && (
              <p className="text-[10px] text-muted-foreground">
                Each band was silenced once. A green bar means removing it lowered the match score, so it supported
                the match.
              </p>
            )}

            {frequencyResult && (
              <div className={staleReason ? "space-y-2 opacity-50 pointer-events-none" : "space-y-2"}>
                <BaselineSimilarityCard result={frequencyResult} />
                <div className="space-y-1" data-testid="saliency-band-list">
                  {frequencyResult.bands.map((band) => {
                    const classification = classifySegment(band);
                    const widthPct = isBandResultBelowDisplayThreshold
                      ? 0
                      : Math.min(1, band.influence_strength / maxBandInfluence) * 100;
                    return (
                      <div
                        key={band.band_index}
                        className="flex items-center gap-2 text-xs"
                        title={`Occluded similarity: ${formatScore(band.occluded_similarity)} · ${CLASSIFICATION_LABEL[classification]}`}
                      >
                        <div className="w-40 shrink-0">
                          <div className="text-sm font-medium">{band.label}</div>
                          <div className="text-xs text-muted-foreground">
                            {formatHzRange(band.low_hz, band.high_hz)}
                          </div>
                        </div>
                        <div className="h-3 flex-1 overflow-hidden rounded bg-muted">
                          <div
                            data-testid={`saliency-band-bar-${band.band_index}`}
                            data-classification={classification}
                            className="h-full rounded"
                            style={{
                              width: `${widthPct}%`,
                              backgroundColor: CLASSIFICATION_UNDERLINE[classification],
                            }}
                          />
                        </div>
                        <span className="w-16 shrink-0 text-right font-mono tabular-nums">
                          {formatSignedShort(band.similarity_change)}
                        </span>
                      </div>
                    );
                  })}
                </div>
                <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
                  {(["supports", "opposes", "minimal"] as const).map((classification) => (
                    <span key={classification} className="flex items-center gap-1">
                      <span
                        className="h-2.5 w-2.5 rounded-sm"
                        style={{ backgroundColor: CLASSIFICATION_UNDERLINE[classification] }}
                      />
                      {CLASSIFICATION_LABEL[classification]}
                    </span>
                  ))}
                </div>
                {isBandResultBelowDisplayThreshold && (
                  <p className="text-[10px] text-muted-foreground">
                    All band changes are below the minimum display-influence threshold.
                  </p>
                )}
              </div>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};

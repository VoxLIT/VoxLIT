import { useEffect, useRef, useState } from "react";
import Plot from "react-plotly.js";
import { AlertCircle, Loader2 } from "lucide-react";
import { API_BASE } from "@/lib/api";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { InfoTooltip } from "./InfoTooltip";

export type SweepPerturbationType = "noise" | "pitch_shift" | "time_stretch";
type SweepDirection = "stronger" | "down" | "up" | "slower" | "faster";

export interface SweepPoint {
  strength: number;
  direction: SweepDirection;
  status: "ok" | "not_applied";
  reason: string | null;
  similarity: number | null;
  same_speaker: boolean | null;
  snr_db: number | null;
}

export interface SweepFlip {
  direction: SweepDirection;
  flipped: boolean;
  flip_strength: number | null;
  last_safe_strength: number | null;
  already_below_at_weakest: boolean;
}

export interface PerturbationSweepResponse {
  model: string;
  model_label: string;
  threshold: number;
  perturbation_type: SweepPerturbationType;
  points: SweepPoint[];
  flips: SweepFlip[];
  summary: string;
  source_recording_id: string;
}

const SWEEP_TYPES: { value: SweepPerturbationType; label: string }[] = [
  { value: "noise", label: "Noise" },
  { value: "pitch_shift", label: "Pitch shift" },
  { value: "time_stretch", label: "Time stretch" },
];

const DIRECTION_LABEL: Record<SweepDirection, string> = {
  stronger: "Added noise",
  down: "Pitch down",
  up: "Pitch up",
  slower: "Slower",
  faster: "Faster",
};

const AXIS_TITLE: Record<SweepPerturbationType, string> = {
  noise: "Noise level (SNR in dB)",
  pitch_shift: "Pitch shift (semitones)",
  time_stretch: "Speed factor",
};

const SAFE_COLOR = "#10b981"; // emerald-500
const FLIPPED_COLOR = "#f43f5e"; // rose-500
const NEUTRAL_COLOR = "#9ca3af"; // gray-400

const formatNumber = (value: number) => Number(value.toPrecision(3)).toString();

function formatStrength(type: SweepPerturbationType, strength: number): string {
  if (type === "pitch_shift") return `${strength > 0 ? "+" : ""}${formatNumber(strength)} semitones`;
  if (type === "time_stretch") return `${formatNumber(strength)}× speed`;
  return `noise level ${formatNumber(strength)}`;
}

function flipLines(type: SweepPerturbationType, flip: SweepFlip): string[] {
  const lines: string[] = [];
  if (flip.last_safe_strength !== null) {
    lines.push(`Safe up to ${formatStrength(type, flip.last_safe_strength)}`);
  }
  if (flip.flipped && flip.flip_strength !== null) {
    lines.push(
      flip.already_below_at_weakest
        ? `Flips at about ${formatStrength(type, flip.flip_strength)} (already below the threshold at the weakest tested strength)`
        : `Flips at about ${formatStrength(type, flip.flip_strength)}`
    );
  } else if (flip.last_safe_strength !== null) {
    lines.push("Never flips in this range");
  } else {
    lines.push("No usable points in this direction");
  }
  return lines;
}

const ANCHOR_SIMILARITY = 1.0;

/** Where the unchanged clip sits on the x axis. Noise has no zero on its log
 *  axis, so its anchor is parked just left of the weakest tested level. */
function anchorStrength(type: SweepPerturbationType, strengths: number[]): number {
  if (type === "pitch_shift") return 0;
  if (type === "time_stretch") return 1;
  return Math.min(...strengths) / 2;
}

function buildTraces(result: PerturbationSweepResponse) {
  const type = result.perturbation_type;
  const okPoints = result.points.filter((point) => point.status === "ok" && point.similarity !== null);
  const skipped = result.points.filter((point) => point.status === "not_applied");
  const strengths = result.points.map((point) => point.strength);
  const similarities = okPoints.map((point) => point.similarity as number);
  // Points with no similarity are parked on the bottom edge of the chart.
  const floor = Math.min(result.threshold, ...similarities) - 0.05;

  const hover = (point: SweepPoint) =>
    [
      formatStrength(type, point.strength),
      point.snr_db !== null ? `SNR ${point.snr_db.toFixed(1)} dB` : null,
      point.similarity !== null ? `Similarity ${point.similarity.toFixed(4)}` : null,
      point.status === "ok"
        ? point.same_speaker
          ? "Same speaker"
          : "Different speaker"
        : `Not applied: ${point.reason ?? "unknown reason"}`,
    ]
      .filter(Boolean)
      .join("<br>");

  const markerTrace = (name: string, points: SweepPoint[], color: string) => ({
    name,
    type: "scatter" as const,
    mode: "markers" as const,
    x: points.map((point) => point.strength),
    y: points.map((point) => point.similarity),
    text: points.map(hover),
    hoverinfo: "text" as const,
    marker: { color, size: 9 },
  });

  const flipped = result.flips.filter((flip) => flip.flipped && flip.flip_strength !== null);

  // Display-only anchor for the unchanged clip: compared with itself it
  // scores 1.0 by definition. It never comes from the backend and plays no
  // part in the flip-point text.
  const anchorX = anchorStrength(type, strengths);
  const anchorLabel = type === "noise" ? "No noise" : "Original (unchanged)";

  // One line per direction, each starting at the anchor and walking outward,
  // so the two directions are never joined to each other.
  const directionLines = result.flips.map((flip) => {
    const points = okPoints
      .filter((point) => point.direction === flip.direction)
      .sort((a, b) => Math.abs(a.strength - anchorX) - Math.abs(b.strength - anchorX));
    return {
      name: DIRECTION_LABEL[flip.direction],
      type: "scatter" as const,
      mode: "lines" as const,
      x: [anchorX, ...points.map((point) => point.strength)],
      y: [ANCHOR_SIMILARITY, ...points.map((point) => point.similarity as number)],
      hoverinfo: "skip" as const,
      line: { color: NEUTRAL_COLOR, width: 1.5 },
    };
  });

  return {
    floor,
    anchorX,
    anchorLabel,
    traces: [
      ...directionLines,
      {
        name: "Threshold",
        type: "scatter" as const,
        mode: "lines" as const,
        x: [Math.min(anchorX, ...strengths), Math.max(anchorX, ...strengths)],
        y: [result.threshold, result.threshold],
        hoverinfo: "skip" as const,
        line: { color: "#6b7280", width: 1.5, dash: "dash" as const },
      },
      markerTrace(
        "Same speaker",
        okPoints.filter((point) => point.same_speaker),
        SAFE_COLOR
      ),
      markerTrace(
        "Different speaker",
        okPoints.filter((point) => !point.same_speaker),
        FLIPPED_COLOR
      ),
      {
        name: "Not applied",
        type: "scatter" as const,
        mode: "markers" as const,
        x: skipped.map((point) => point.strength),
        y: skipped.map(() => floor),
        text: skipped.map(hover),
        hoverinfo: "text" as const,
        marker: { color: NEUTRAL_COLOR, size: 9, symbol: "circle-open" as const, line: { width: 2 } },
      },
      {
        name: anchorLabel,
        type: "scatter" as const,
        mode: "markers+text" as const,
        x: [anchorX],
        y: [ANCHOR_SIMILARITY],
        text: [anchorLabel],
        textposition: "top center" as const,
        hoverinfo: "text" as const,
        marker: { color: NEUTRAL_COLOR, size: 10, symbol: "circle-open" as const, line: { width: 2 } },
      },
      {
        name: "Flip point",
        type: "scatter" as const,
        mode: "markers" as const,
        x: flipped.map((flip) => flip.flip_strength),
        y: flipped.map(() => result.threshold),
        text: flipped.map(
          (flip) => `${DIRECTION_LABEL[flip.direction]}: flips at about ${formatStrength(type, flip.flip_strength as number)}`
        ),
        hoverinfo: "text" as const,
        marker: { color: "#111827", size: 13, symbol: "diamond-open" as const, line: { width: 2 } },
      },
    ],
  };
}

interface PerturbationSweepCardProps {
  model: string;
  recordingId: string | null;
  recordingLabel: string | null;
}

export const PerturbationSweepCard = ({ model, recordingId, recordingLabel }: PerturbationSweepCardProps) => {
  const [sweepType, setSweepType] = useState<SweepPerturbationType>("noise");
  const [result, setResult] = useState<PerturbationSweepResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  // A sweep belongs to one recording and one model: drop it (and any
  // in-flight request) as soon as either changes.
  useEffect(() => {
    abortRef.current?.abort();
    abortRef.current = null;
    setResult(null);
    setError(null);
    setIsRunning(false);
  }, [model, recordingId]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const runSweep = async () => {
    if (!recordingId) return;
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setIsRunning(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/tasks/verification/perturbation/sweep`, {
        method: "POST",
        credentials: "include",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model, recording_id: recordingId, perturbation_type: sweepType }),
        signal: controller.signal,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(
          typeof payload.detail === "string" ? payload.detail : `Perturbation sweep failed (${response.status}).`
        );
      }
      if (abortRef.current !== controller) return;
      setResult(payload as PerturbationSweepResponse);
    } catch (caught) {
      if (caught instanceof DOMException && caught.name === "AbortError") return;
      if (abortRef.current !== controller) return;
      setError(caught instanceof Error ? caught.message : "Perturbation sweep failed.");
    } finally {
      if (abortRef.current === controller) {
        abortRef.current = null;
        setIsRunning(false);
      }
    }
  };

  const handleTypeChange = (value: string) => {
    // Radix emits "" when the active item is clicked again -- keep the type.
    if (value !== "noise" && value !== "pitch_shift" && value !== "time_stretch") return;
    setSweepType(value);
  };

  const chart = result ? buildTraces(result) : null;
  const useLogAxis = result ? result.perturbation_type !== "pitch_shift" : false;
  const notAppliedCount = result ? result.points.filter((point) => point.status === "not_applied").length : 0;

  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-sm">
          Perturbation sweep
          <InfoTooltip title="Perturbation sweep">
            <p>
              Applies one kind of perturbation at a fixed series of strengths and compares each perturbed copy with
              the original clip.
            </p>
            <p>The flip point is where the similarity first drops below the model&apos;s threshold.</p>
          </InfoTooltip>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="text-xs text-muted-foreground">
          Selected recording:{" "}
          <span className="font-medium text-foreground">{recordingId ? (recordingLabel ?? recordingId) : "none"}</span>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <ToggleGroup
            type="single"
            size="sm"
            variant="outline"
            value={sweepType}
            onValueChange={handleTypeChange}
            aria-label="Sweep perturbation type"
          >
            {SWEEP_TYPES.map((option) => (
              <ToggleGroupItem key={option.value} value={option.value} className="h-7 px-2 text-xs">
                {option.label}
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <div className="flex-1" />
          <Button size="sm" variant="outline" onClick={runSweep} disabled={!recordingId || isRunning}>
            {isRunning ? (
              <>
                <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
                Running sweep…
              </>
            ) : (
              "Run sweep"
            )}
          </Button>
        </div>

        {!recordingId && (
          <p className="text-xs text-muted-foreground">Select a recording to run a sweep.</p>
        )}

        <p className="text-[10px] text-muted-foreground">
          Each point is the same clip, perturbed more strongly, compared with its original. Below the dashed line the
          model would no longer say it is the same speaker.
        </p>

        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Sweep could not be completed</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {result && chart && (
          <div className="space-y-3">
            <div className="space-y-2 rounded-md border p-2.5" data-testid="sweep-summary">
              <p className="text-xs">{result.summary}</p>
              <div className="space-y-1">
                {result.flips.map((flip) => (
                  <div key={flip.direction} className="flex flex-wrap items-baseline gap-x-2 text-xs">
                    <span className="font-medium">{DIRECTION_LABEL[flip.direction]}</span>
                    {flipLines(result.perturbation_type, flip).map((line) => (
                      <span
                        key={line}
                        className={line.startsWith("Flips") ? "text-rose-700" : "text-muted-foreground"}
                      >
                        {line}
                      </span>
                    ))}
                  </div>
                ))}
              </div>
              {notAppliedCount > 0 && (
                <p className="text-[10px] text-muted-foreground">
                  {notAppliedCount} point{notAppliedCount > 1 ? "s" : ""} could not be applied (grey hollow markers)
                  and {notAppliedCount > 1 ? "are" : "is"} left out of the analysis.
                </p>
              )}
            </div>

            <div className="h-72 w-full">
              <Plot
                data={chart.traces}
                layout={{
                  autosize: true,
                  margin: { l: 50, r: 16, t: 10, b: 60 },
                  showlegend: false,
                  hovermode: "closest",
                  plot_bgcolor: "transparent",
                  paper_bgcolor: "transparent",
                  font: { size: 10 },
                  xaxis: {
                    title: { text: AXIS_TITLE[result.perturbation_type] },
                    type: useLogAxis ? "log" : "linear",
                    tickmode: "array",
                    tickvals: [chart.anchorX, ...result.points.map((point) => point.strength)],
                    ticktext: [
                      result.perturbation_type === "noise" ? chart.anchorLabel : formatNumber(chart.anchorX),
                      ...result.points.map((point) =>
                        point.snr_db !== null
                          ? `${formatNumber(point.strength)}<br>${point.snr_db.toFixed(0)} dB`
                          : formatNumber(point.strength)
                      ),
                    ],
                  },
                  yaxis: {
                    title: { text: "Cosine similarity" },
                    // Headroom above 1.0 for the anchor's label.
                    range: [chart.floor - 0.03, 1.12],
                  },
                }}
                config={{ displayModeBar: false, responsive: true }}
                style={{ width: "100%", height: "100%" }}
                useResizeHandler
              />
            </div>
            {result.perturbation_type === "pitch_shift" && (
              <p className="text-[10px] text-muted-foreground">
                Even small pitch shifts lower the score, partly because the pitch-shift algorithm also adds slight
                processing artifacts.
              </p>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

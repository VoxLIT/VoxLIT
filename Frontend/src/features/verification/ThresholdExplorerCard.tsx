import { useEffect, useMemo, useState } from "react";
import Plot from "react-plotly.js";
import { SlidersHorizontal } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { InfoTooltip } from "./InfoTooltip";
import type { BatchAnalysisResponse } from "./batchTypes";
import {
  buildPairs,
  changedPairs,
  equalErrorRate,
  errorCurve,
  metricsAt,
  similarityRange,
} from "./thresholdMetrics";

interface ThresholdExplorerCardProps {
  batchResult: BatchAnalysisResponse;
  resolveLabel?: (id: string) => string;
}

const SAME_COLOR = "#10b981"; // emerald-500
const DIFFERENT_COLOR = "#f43f5e"; // rose-500
const NEUTRAL_COLOR = "#9ca3af"; // gray-400
const LINE_COLOR = "#111827"; // gray-900
const CALIBRATED_COLOR = "#6b7280"; // gray-500

const SLIDER_STEP = 0.01;
const CURVE_STEPS = 101;
const HISTOGRAM_BINS = 30;
const MAX_CHANGED_ROWS = 10;

const formatScore = (value: number) => value.toFixed(4);
const formatCount = (value: number) => value.toLocaleString();
const formatPercent = (value: number | null) => (value === null ? "n/a" : `${(value * 100).toFixed(1)}%`);
const decisionLabel = (accepted: boolean) => (accepted ? "Same speaker" : "Different speakers");

const verticalLine = (x: number, dash: "dash" | "solid", color: string) => ({
  type: "line" as const,
  xref: "x" as const,
  yref: "paper" as const,
  x0: x,
  x1: x,
  y0: 0,
  y1: 1,
  line: { color, width: 2, dash },
});

const BASE_LAYOUT = {
  autosize: true,
  margin: { l: 50, r: 16, t: 10, b: 40 },
  plot_bgcolor: "transparent",
  paper_bgcolor: "transparent",
  font: { size: 10 },
  showlegend: true,
  legend: { orientation: "h" as const, y: -0.3 },
};

/**
 * What-if view of the pair threshold. Everything here is derived locally from
 * the batch result; the slider never changes the result, its decisions, the
 * clustering or any other card.
 */
export const ThresholdExplorerCard = ({ batchResult, resolveLabel }: ThresholdExplorerCardProps) => {
  const resolve = resolveLabel ?? ((id: string) => id);
  const calibrated = batchResult.threshold;
  const [whatIf, setWhatIf] = useState(calibrated);

  // A new batch result starts again from its own calibrated threshold.
  useEffect(() => {
    setWhatIf(batchResult.threshold);
  }, [batchResult]);

  const pairs = useMemo(() => buildPairs(batchResult), [batchResult]);
  const hasGroundTruth = useMemo(() => pairs.some((pair) => pair.sameSpeaker !== null), [pairs]);
  const curve = useMemo(() => (hasGroundTruth ? errorCurve(pairs, CURVE_STEPS) : []), [pairs, hasGroundTruth]);
  const eer = useMemo(() => (hasGroundTruth ? equalErrorRate(pairs) : null), [pairs, hasGroundTruth]);
  const metrics = useMemo(() => metricsAt(pairs, whatIf), [pairs, whatIf]);
  const changed = useMemo(() => changedPairs(pairs, calibrated, whatIf), [pairs, calibrated, whatIf]);

  // Slider bounds snap outward to the step grid, and always include the
  // calibrated threshold so the slider can start (and reset) there.
  const range = similarityRange(pairs) ?? { min: calibrated, max: calibrated };
  const toSteps = (value: number) => Number((value / SLIDER_STEP).toFixed(6));
  const sliderMin = Number((Math.floor(toSteps(Math.min(range.min, calibrated))) * SLIDER_STEP).toFixed(2));
  const sliderMax = Number((Math.ceil(toSteps(Math.max(range.max, calibrated))) * SLIDER_STEP).toFixed(2));

  const bins = {
    start: sliderMin,
    end: sliderMax + SLIDER_STEP,
    size: Math.max((sliderMax - sliderMin) / HISTOGRAM_BINS, SLIDER_STEP),
  };
  const histogram = (name: string, values: number[], color: string) => ({
    name,
    type: "histogram" as const,
    x: values,
    xbins: bins,
    opacity: 0.6,
    marker: { color },
  });
  const histogramTraces = hasGroundTruth
    ? [
        histogram(
          "Same speaker",
          pairs.filter((pair) => pair.sameSpeaker === true).map((pair) => pair.similarity),
          SAME_COLOR
        ),
        histogram(
          "Different speaker",
          pairs.filter((pair) => pair.sameSpeaker === false).map((pair) => pair.similarity),
          DIFFERENT_COLOR
        ),
      ]
    : [
        histogram(
          "All pairs",
          pairs.map((pair) => pair.similarity),
          NEUTRAL_COLOR
        ),
      ];

  const thresholdShapes = [
    verticalLine(calibrated, "dash", CALIBRATED_COLOR),
    verticalLine(whatIf, "solid", LINE_COLOR),
  ];

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm">
          <SlidersHorizontal className="h-4 w-4" /> Threshold explorer
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3 text-xs">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="flex items-center gap-1 text-muted-foreground">
              What-if threshold
              <Badge variant="secondary" data-testid="whatif-threshold">
                {formatScore(whatIf)}
              </Badge>
            </span>
            <span className="flex items-center gap-1 text-muted-foreground">
              Calibrated ({batchResult.model_label})
              <Badge variant="secondary" data-testid="calibrated-threshold">
                {formatScore(calibrated)}
              </Badge>
              <InfoTooltip title="Calibrated threshold">
                <p>
                  The calibrated threshold was chosen on a separate held-out split, so it will not exactly match this
                  batch&apos;s EER threshold. This slider is a what-if view only.
                </p>
              </InfoTooltip>
            </span>
          </div>
          <div className="flex items-center gap-3">
            <input
              type="range"
              aria-label="What-if threshold"
              className="h-2 flex-1 cursor-pointer accent-primary"
              min={sliderMin}
              max={sliderMax}
              step={SLIDER_STEP}
              value={whatIf}
              onChange={(event) => setWhatIf(Number(event.target.value))}
            />
            <Button size="sm" variant="outline" onClick={() => setWhatIf(calibrated)} disabled={whatIf === calibrated}>
              Reset to calibrated
            </Button>
          </div>
        </div>

        <div className="space-y-1">
          <span className="font-medium">Score distribution</span>
          <div className="h-56 w-full">
            <Plot
              data={histogramTraces}
              layout={{
                ...BASE_LAYOUT,
                barmode: "overlay",
                xaxis: { title: { text: "Cosine similarity" } },
                yaxis: { title: { text: "Pairs" } },
                shapes: thresholdShapes,
              }}
              config={{ displayModeBar: false, responsive: true }}
              style={{ width: "100%", height: "100%" }}
              useResizeHandler
            />
          </div>
          <p className="text-[10px] text-muted-foreground">
            Dashed line: calibrated threshold. Solid line: what-if threshold.
          </p>
        </div>

        {hasGroundTruth ? (
          <>
            <div className="space-y-1">
              <span className="font-medium">Error rates</span>
              <div className="h-56 w-full">
                <Plot
                  data={[
                    {
                      name: "FAR",
                      type: "scatter" as const,
                      mode: "lines" as const,
                      x: curve.map((point) => point.threshold),
                      y: curve.map((point) => point.far),
                      line: { color: DIFFERENT_COLOR, width: 2 },
                    },
                    {
                      name: "FRR",
                      type: "scatter" as const,
                      mode: "lines" as const,
                      x: curve.map((point) => point.threshold),
                      y: curve.map((point) => point.frr),
                      line: { color: SAME_COLOR, width: 2 },
                    },
                    {
                      name: "EER",
                      type: "scatter" as const,
                      mode: "markers" as const,
                      x: eer ? [eer.threshold] : [],
                      y: eer ? [eer.eer] : [],
                      text: eer ? [`EER ${formatPercent(eer.eer)} at threshold ${formatScore(eer.threshold)}`] : [],
                      hoverinfo: "text" as const,
                      marker: { color: LINE_COLOR, size: 11, symbol: "diamond-open" as const, line: { width: 2 } },
                    },
                  ]}
                  layout={{
                    ...BASE_LAYOUT,
                    xaxis: { title: { text: "Threshold" }, range: [sliderMin, sliderMax] },
                    yaxis: { title: { text: "Error rate" }, range: [-0.03, 1.03] },
                    shapes: [verticalLine(whatIf, "solid", LINE_COLOR)],
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-1 text-muted-foreground" data-testid="threshold-stats">
              <span>
                Accepted pairs: {formatCount(metrics.accepted)} of {formatCount(metrics.totalPairs)}
              </span>
              <span>Accuracy: {formatPercent(metrics.accuracy)}</span>
              <span className="flex items-center gap-1">
                False accepts: {formatCount(metrics.falseAccepts ?? 0)} (FAR {formatPercent(metrics.far)})
                <InfoTooltip title="FAR (false accept rate)">
                  <p>Different speakers wrongly accepted as the same person.</p>
                </InfoTooltip>
              </span>
              <span className="flex items-center gap-1">
                False rejects: {formatCount(metrics.falseRejects ?? 0)} (FRR {formatPercent(metrics.frr)})
                <InfoTooltip title="FRR (false reject rate)">
                  <p>The same speaker wrongly rejected.</p>
                </InfoTooltip>
              </span>
              <span className="col-span-2 flex items-center gap-1">
                {eer
                  ? `EER ${formatPercent(eer.eer)} at threshold ${formatScore(eer.threshold)}`
                  : "EER: n/a (needs both same-speaker and different-speaker pairs)"}
                <InfoTooltip title="EER (equal error rate)">
                  <p>
                    The threshold where both error rates are equal. A common single-number summary of a verification
                    model.
                  </p>
                </InfoTooltip>
              </span>
            </div>
          </>
        ) : (
          <>
            <div className="text-muted-foreground" data-testid="threshold-stats">
              Accepted pairs: {formatCount(metrics.accepted)} of {formatCount(metrics.totalPairs)}
            </div>
            <p className="text-muted-foreground">
              Error rates need known speaker groups, which this dataset does not have.
            </p>
          </>
        )}

        {hasGroundTruth && (
          <div className="space-y-1" data-testid="changed-pairs">
            <span className="font-medium">Decisions that change</span>
            {changed.length === 0 ? (
              <p className="text-muted-foreground">No decisions change</p>
            ) : (
              <>
                <ul className="space-y-1">
                  {changed.slice(0, MAX_CHANGED_ROWS).map((pair) => {
                    const labelA = resolve(batchResult.labels[pair.i]);
                    const labelB = resolve(batchResult.labels[pair.j]);
                    return (
                      <li key={`${pair.i}-${pair.j}`} className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                        <span className="truncate font-medium" title={`${labelA} ↔ ${labelB}`}>
                          {labelA} ↔ {labelB}
                        </span>
                        <Badge variant="secondary">{formatScore(pair.similarity)}</Badge>
                        <span className="text-muted-foreground">
                          {decisionLabel(pair.acceptedAtCalibrated)} →{" "}
                          <span className={pair.acceptedAtWhatIf ? "text-emerald-700" : "text-rose-700"}>
                            {decisionLabel(pair.acceptedAtWhatIf)}
                          </span>
                        </span>
                      </li>
                    );
                  })}
                </ul>
                {changed.length > MAX_CHANGED_ROWS && (
                  <p className="text-[10px] text-muted-foreground">
                    Showing {MAX_CHANGED_ROWS} of {formatCount(changed.length)} changed pairs.
                  </p>
                )}
              </>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
};

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
  /** Batch indices of a box/lasso group on the embedding plot. Three or more
   *  enable Selection mode; anything else leaves the card on All pairs. */
  selectedIndices?: number[];
}

const MIN_SELECTION_SIZE = 3;
const SMALL_CLASS_PAIRS = 10;

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

// The slider and both charts share one x range and one pair of horizontal
// plot margins (px), so the thumb sits directly above the charts' threshold
// lines.
const PLOT_MARGIN_LEFT = 56;
const PLOT_MARGIN_RIGHT = 16;
const SLIDER_THUMB_WIDTH = 16;
const SLIDER_TRACK_HEIGHT = 4;
const SLIDER_CLASS = "sv-threshold-slider";

// A native range thumb only travels between half a thumb width from each end
// of the input, so the input is widened by half a thumb on each side: the
// thumb centre then lands exactly on the plot-area edges at min and max. The
// thumb is sized explicitly so this holds in every browser.
const thumbCss = `
  box-sizing: border-box;
  width: ${SLIDER_THUMB_WIDTH}px;
  height: ${SLIDER_THUMB_WIDTH}px;
  border: 2px solid #ffffff;
  border-radius: 9999px;
  background: ${LINE_COLOR};
  cursor: pointer;`;
const trackCss = `
  height: ${SLIDER_TRACK_HEIGHT}px;
  border-radius: 9999px;
  background: ${NEUTRAL_COLOR};`;
const SLIDER_CSS = `
.${SLIDER_CLASS} {
  -webkit-appearance: none;
  appearance: none;
  display: block;
  box-sizing: border-box;
  width: calc(100% + ${SLIDER_THUMB_WIDTH}px);
  height: ${SLIDER_THUMB_WIDTH}px;
  margin: 0 -${SLIDER_THUMB_WIDTH / 2}px;
  padding: 0;
  border: 0;
  background: transparent;
  cursor: pointer;
}
.${SLIDER_CLASS}::-webkit-slider-runnable-track {${trackCss}
}
.${SLIDER_CLASS}::-moz-range-track {${trackCss}
}
.${SLIDER_CLASS}::-webkit-slider-thumb {
  -webkit-appearance: none;
  appearance: none;
  margin-top: ${(SLIDER_TRACK_HEIGHT - SLIDER_THUMB_WIDTH) / 2}px;${thumbCss}
}
.${SLIDER_CLASS}::-moz-range-thumb {${thumbCss}
}
`;

const BASE_LAYOUT = {
  autosize: true,
  margin: { l: PLOT_MARGIN_LEFT, r: PLOT_MARGIN_RIGHT, t: 10, b: 40 },
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
export const ThresholdExplorerCard = ({ batchResult, resolveLabel, selectedIndices }: ThresholdExplorerCardProps) => {
  const resolve = resolveLabel ?? ((id: string) => id);
  const calibrated = batchResult.threshold;
  const [whatIf, setWhatIf] = useState(calibrated);

  // A new batch result starts again from its own calibrated threshold.
  useEffect(() => {
    setWhatIf(batchResult.threshold);
  }, [batchResult]);

  // Unique, in-range, sorted selection as a string, so a re-rendered parent
  // handing over an equal array is not treated as a new selection.
  const recordingCount = batchResult.similarity_matrix.length;
  const selectionKey = useMemo(() => {
    const valid = new Set(
      (selectedIndices ?? []).filter((index) => Number.isInteger(index) && index >= 0 && index < recordingCount)
    );
    return valid.size >= MIN_SELECTION_SIZE ? [...valid].sort((a, b) => a - b).join(",") : "";
  }, [selectedIndices, recordingCount]);
  const selection = useMemo(() => (selectionKey ? selectionKey.split(",").map(Number) : []), [selectionKey]);
  const hasSelection = selection.length > 0;

  // A new selection switches to Selection; clearing it falls back to All pairs.
  const [mode, setMode] = useState<"all" | "selection">(hasSelection ? "selection" : "all");
  useEffect(() => {
    setMode(selectionKey ? "selection" : "all");
  }, [selectionKey]);
  const selectionMode = hasSelection && mode === "selection";

  const allPairs = useMemo(() => buildPairs(batchResult), [batchResult]);
  const selectedPairs = useMemo(() => buildPairs(batchResult, selection), [batchResult, selection]);
  const pairs = selectionMode ? selectedPairs : allPairs;
  const hasGroundTruth = useMemo(() => allPairs.some((pair) => pair.sameSpeaker !== null), [allPairs]);
  const rawMetrics = useMemo(() => metricsAt(pairs, whatIf), [pairs, whatIf]);
  const changed = useMemo(() => changedPairs(pairs, calibrated, whatIf), [pairs, calibrated, whatIf]);

  // Selection-only warnings. A selection missing one class reports no rates at all.
  const sameCount = rawMetrics.samePairs ?? 0;
  const differentCount = rawMetrics.differentPairs ?? 0;
  const singleClass = selectionMode && hasGroundTruth && (sameCount === 0 || differentCount === 0);
  const smallSelection =
    selectionMode &&
    hasGroundTruth &&
    !singleClass &&
    (sameCount < SMALL_CLASS_PAIRS || differentCount < SMALL_CLASS_PAIRS);
  const metrics = singleClass
    ? { ...rawMetrics, far: null, frr: null, accuracy: null, balancedAccuracy: null }
    : rawMetrics;
  const curve = useMemo(
    () => (hasGroundTruth && !singleClass ? errorCurve(pairs, CURVE_STEPS) : []),
    [pairs, hasGroundTruth, singleClass]
  );
  const eer = useMemo(() => (hasGroundTruth ? equalErrorRate(pairs) : null), [pairs, hasGroundTruth]);

  // Slider bounds snap outward to the step grid, and always include the
  // calibrated threshold so the slider can start (and reset) there. They come
  // from every pair in the batch, so switching modes never moves the x range.
  const range = similarityRange(allPairs) ?? { min: calibrated, max: calibrated };
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
          selectionMode ? "Selected pairs" : "All pairs",
          pairs.map((pair) => pair.similarity),
          NEUTRAL_COLOR
        ),
      ];

  // One fixed x range for the slider and both charts.
  const sharedXAxis = { range: [sliderMin, sliderMax], autorange: false as const, automargin: false };

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
        <style>{SLIDER_CSS}</style>
        {hasSelection && (
          <div className="flex items-center gap-1" role="group" aria-label="Pairs to analyse">
            <Button
              size="sm"
              variant={selectionMode ? "outline" : "default"}
              aria-pressed={!selectionMode}
              onClick={() => setMode("all")}
            >
              All pairs
            </Button>
            <Button
              size="sm"
              variant={selectionMode ? "default" : "outline"}
              aria-pressed={selectionMode}
              onClick={() => setMode("selection")}
            >
              Selection
            </Button>
          </div>
        )}
        {selectionMode ? (
          <div className="space-y-1">
            <p className="font-medium" data-testid="selection-note">
              Selection: {formatCount(selection.length)} clips · {formatCount(pairs.length)} pairs
              {hasGroundTruth &&
                ` (${formatCount(sameCount)} same-speaker · ${formatCount(differentCount)} different-speaker)`}
            </p>
            {smallSelection && (
              <p className="text-amber-700" data-testid="selection-warning">
                Small selection: rates are based on few pairs and can swing a lot.
              </p>
            )}
            {singleClass && (
              <p className="text-amber-700" data-testid="selection-warning">
                This selection has no same-speaker (or no different-speaker) pairs, so error rates can&apos;t be
                computed.
              </p>
            )}
          </div>
        ) : (
          <p className="text-muted-foreground" data-testid="selection-hint">
            Tip: use Box or Lasso on the embedding plot to analyse a group of clips.
          </p>
        )}
        <div className="space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2" data-testid="threshold-header-row">
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
              <Button
                size="sm"
                variant="outline"
                className="ml-1"
                onClick={() => setWhatIf(calibrated)}
                disabled={whatIf === calibrated}
              >
                Reset to calibrated
              </Button>
            </span>
          </div>
          <div
            data-testid="threshold-slider-row"
            style={{ paddingLeft: PLOT_MARGIN_LEFT, paddingRight: PLOT_MARGIN_RIGHT }}
          >
            <input
              type="range"
              aria-label="What-if threshold"
              className={SLIDER_CLASS}
              min={sliderMin}
              max={sliderMax}
              step={SLIDER_STEP}
              value={whatIf}
              onChange={(event) => setWhatIf(Number(event.target.value))}
            />
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
                xaxis: { title: { text: "Cosine similarity" }, ...sharedXAxis },
                yaxis: { title: { text: "Pairs" }, automargin: false },
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
                    xaxis: { title: { text: "Threshold" }, ...sharedXAxis },
                    yaxis: { title: { text: "Error rate" }, range: [-0.03, 1.03], automargin: false },
                    shapes: [verticalLine(whatIf, "solid", LINE_COLOR)],
                  }}
                  config={{ displayModeBar: false, responsive: true }}
                  style={{ width: "100%", height: "100%" }}
                  useResizeHandler
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-1 text-muted-foreground" data-testid="threshold-stats">
              <span className="font-medium text-foreground">
                Balanced accuracy: {formatPercent(metrics.balancedAccuracy)}
              </span>
              <span className="flex items-center gap-1">
                Accuracy: {formatPercent(metrics.accuracy)}
                <InfoTooltip title="Accuracy">
                  <p>
                    Share of all pairs decided correctly. Most pairs are different-speaker pairs, so rejecting
                    everything still scores high. Balanced accuracy weights both kinds of pair equally.
                  </p>
                </InfoTooltip>
              </span>
              <span className="col-span-2">
                Accepted pairs: {formatCount(metrics.accepted)} of {formatCount(metrics.totalPairs)}
              </span>
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
            <p className="text-[10px] text-muted-foreground" data-testid="class-counts">
              {formatCount(metrics.samePairs ?? 0)} same-speaker pairs · {formatCount(metrics.differentPairs ?? 0)}{" "}
              different-speaker pairs
            </p>
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

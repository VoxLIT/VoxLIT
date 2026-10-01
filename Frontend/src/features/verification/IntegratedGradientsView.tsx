import { useMemo } from "react";
import Plot from "react-plotly.js";
import { AlertTriangle, Check } from "lucide-react";
import { FrequencyBandGuideTooltip, formatBandLabel } from "./frequencyBands";
import type { IntegratedGradientsSaliencyResponse } from "./saliencyTypes";

const SUPPORTS_COLOR = "#10b981"; // emerald-500
const OPPOSES_COLOR = "#f43f5e"; // rose-500
const NEUTRAL_COLOR = "#9ca3af"; // gray-400

// Diverging and symmetric around 0: red opposes, white is neutral, green supports.
const DIVERGING_COLORSCALE: [number, string][] = [
  [0, OPPOSES_COLOR],
  [0.5, "#ffffff"],
  [1, SUPPORTS_COLOR],
];

const MAX_FREQUENCY_TICKS = 8;

const formatHzRange = (lowHz: number, highHz: number) => `${Math.round(lowHz)}–${Math.round(highHz)} Hz`;
const formatSignedShort = (value: number) => `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
const signColor = (value: number) => (value > 0 ? SUPPORTS_COLOR : value < 0 ? OPPOSES_COLOR : NEUTRAL_COLOR);
const signName = (value: number) => (value > 0 ? "supports" : value < 0 ? "opposes" : "neutral");
const maxAbs = (values: number[]) => values.reduce((max, value) => Math.max(max, Math.abs(value)), 0);

/** Time x frequency Integrated Gradients attribution map with its per-time
 *  and per-band totals. */
export const IntegratedGradientsView = ({ result }: { result: IntegratedGradientsSaliencyResponse }) => {
  const heatmap = useMemo(() => {
    const timeEdges = result.time_edges_seconds;
    const melEdges = result.mel_edges_hz;
    const rowCount = result.attributions.length;
    const rowLabels = melEdges.slice(0, -1).map((low, index) => formatHzRange(low, melEdges[index + 1]));
    const tickStep = Math.max(1, Math.ceil(rowCount / MAX_FREQUENCY_TICKS));
    const tickRows = Array.from({ length: rowCount }, (_, index) => index).filter((index) => index % tickStep === 0);
    return {
      x: timeEdges.slice(0, -1).map((start, index) => (start + timeEdges[index + 1]) / 2),
      // Rows are evenly spaced on the mel scale; only the labels are in Hz.
      y: Array.from({ length: rowCount }, (_, index) => index),
      rowLabels: result.attributions.map((row, index) => row.map(() => rowLabels[index])),
      tickRows,
      tickText: tickRows.map((index) => String(Math.round((melEdges[index] + melEdges[index + 1]) / 2))),
      // Never 0, so an all-zero map still gets a valid symmetric range.
      limit: maxAbs(result.attributions.flat()) || 1,
    };
  }, [result]);

  const maxTimeTotal = maxAbs(result.time_totals);
  const maxBandTotal = maxAbs(result.bands.map((band) => band.total_attribution));

  return (
    <div className="space-y-3">
      <div
        data-testid="ig-completeness-badge"
        data-completeness={result.completeness_ok ? "ok" : "rough"}
        title={`Attributions sum to ${formatSignedShort(result.total_attribution)}; the score changed by ${formatSignedShort(result.expected_total)} from silence (${result.n_steps} steps).`}
        className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10px] font-medium ${
          result.completeness_ok
            ? "border-emerald-300 bg-emerald-50 text-emerald-700"
            : "border-amber-300 bg-amber-50 text-amber-700"
        }`}
      >
        {result.completeness_ok ? (
          <>
            Attributions add up to the score change <Check className="h-3 w-3" aria-hidden />
          </>
        ) : (
          <>
            <AlertTriangle className="h-3 w-3" aria-hidden /> Approximation is rough (n_steps too low)
          </>
        )}
      </div>

      <div className="h-64 w-full" data-testid="ig-heatmap">
        <Plot
          data={[
            {
              type: "heatmap",
              x: heatmap.x,
              y: heatmap.y,
              z: result.attributions,
              customdata: heatmap.rowLabels,
              colorscale: DIVERGING_COLORSCALE,
              zmin: -heatmap.limit,
              zmax: heatmap.limit,
              zmid: 0,
              colorbar: { thickness: 10, len: 0.9, title: { text: "Attribution", side: "right" } },
              hovertemplate: "%{x:.2f} s · %{customdata}<br>Attribution: %{z:+.4f}<extra></extra>",
            },
          ]}
          layout={{
            autosize: true,
            margin: { l: 50, r: 10, t: 10, b: 36 },
            plot_bgcolor: "transparent",
            paper_bgcolor: "transparent",
            font: { size: 10 },
            xaxis: { title: { text: "Time (s)" } },
            yaxis: {
              title: { text: "Frequency (Hz)" },
              tickmode: "array",
              tickvals: heatmap.tickRows,
              ticktext: heatmap.tickText,
            },
          }}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height: "100%" }}
          useResizeHandler
        />
      </div>

      <div className="space-y-1">
        <div className="text-xs font-medium">Total over time</div>
        <div
          className="relative flex h-10 w-full overflow-hidden rounded border border-border"
          data-testid="ig-time-strip"
        >
          <div className="pointer-events-none absolute inset-x-0 top-1/2 border-t border-border" />
          {result.time_totals.map((total, index) => {
            const heightPct = maxTimeTotal > 0 ? (Math.abs(total) / maxTimeTotal) * 50 : 0;
            return (
              <div
                key={index}
                className="relative h-full flex-1"
                title={`${result.time_edges_seconds[index].toFixed(2)}–${result.time_edges_seconds[index + 1].toFixed(2)} s: ${formatSignedShort(total)}`}
              >
                <div
                  data-testid={`ig-time-bar-${index}`}
                  data-direction={signName(total)}
                  className="absolute inset-x-0"
                  style={{
                    height: `${heightPct}%`,
                    backgroundColor: signColor(total),
                    // Supporting totals rise from the centre line, opposing ones hang below it.
                    ...(total >= 0 ? { bottom: "50%" } : { top: "50%" }),
                  }}
                />
              </div>
            );
          })}
        </div>
      </div>

      <div className="space-y-1">
        <div className="flex items-center gap-1 text-xs font-medium">
          Total per frequency band
          <FrequencyBandGuideTooltip />
        </div>
        <div className="space-y-1" data-testid="ig-band-list">
          {result.bands.map((band) => (
            <div key={band.band_index} className="flex items-center gap-2 text-xs">
              <div className="w-44 shrink-0 text-xs font-medium tabular-nums">
                {formatBandLabel(band.band_index, band.low_hz, band.high_hz)}
              </div>
              <div className="h-3 flex-1 overflow-hidden rounded bg-muted">
                <div
                  data-testid={`ig-band-bar-${band.band_index}`}
                  data-direction={signName(band.total_attribution)}
                  className="h-full rounded"
                  style={{
                    width: `${maxBandTotal > 0 ? (Math.abs(band.total_attribution) / maxBandTotal) * 100 : 0}%`,
                    backgroundColor: signColor(band.total_attribution),
                  }}
                />
              </div>
              <span className="w-16 shrink-0 text-right font-mono tabular-nums">
                {formatSignedShort(band.total_attribution)}
              </span>
            </div>
          ))}
        </div>
      </div>

      <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: SUPPORTS_COLOR }} />
          Supports the match
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: OPPOSES_COLOR }} />
          Opposes the match
        </span>
        <span className="flex items-center gap-1">
          <span className="h-2.5 w-2.5 rounded-sm border border-border bg-white" />
          Neutral
        </span>
      </div>
    </div>
  );
};

import { useMemo } from "react";
import Plot from "react-plotly.js";
import { BONAFIDE_COLOR, SPOOF_COLOR } from "./embeddingColors";
import type { EmbeddingRecording } from "./types";

interface EmbeddingScatterProps {
  recordings: EmbeddingRecording[];
  /** Index-aligned with `recordings`; 2 or 3 columns. */
  coordinates: number[][];
  is3D: boolean;
  /** Changes whenever the projection itself changes, so the camera resets then
   *  but not when only the selected point changes. */
  viewKey: string;
  selectedRecordingId: string;
  onSelectRecording: (recordingId: string) => void;
}

const SELECTED_COLOR = "#FFD700";

/**
 * The 2D / 3D scatter itself: one point per recording, coloured by the
 * detector's own spoof score. Purely presentational — the card above it owns
 * the request. Clicking a point reports that recording's id.
 */
export const EmbeddingScatter = ({
  recordings,
  coordinates,
  is3D,
  viewKey,
  selectedRecordingId,
  onSelectRecording,
}: EmbeddingScatterProps) => {
  const data = useMemo(() => {
    const axis = (index: number) => coordinates.map((row) => row[index] ?? 0);
    const dimensions = is3D ? { x: axis(0), y: axis(1), z: axis(2) } : { x: axis(0), y: axis(1) };
    const type = is3D ? "scatter3d" : "scatter";

    const main = {
      type,
      mode: "markers",
      name: "recordings",
      ...dimensions,
      customdata: recordings.map((recording) => [
        recording.recording_id,
        recording.display_filename,
        recording.spoof_probability,
        recording.decision === "spoof" ? "SPOOF" : "BONA FIDE",
      ]),
      hovertemplate:
        "%{customdata[1]}<br>spoof score %{customdata[2]:.3f} • %{customdata[3]}<extra></extra>",
      marker: {
        size: is3D ? 4 : 8,
        opacity: 0.9,
        color: recordings.map((recording) => recording.spoof_probability),
        colorscale: [
          [0, BONAFIDE_COLOR],
          [1, SPOOF_COLOR],
        ],
        cmin: 0,
        cmax: 1,
        // The panel's gradient legend already explains the colour, and a
        // colourbar would cost this narrow column a fifth of its width.
        showscale: false,
        line: { width: 0.5, color: "white" },
      },
    };

    const selectedIndex = recordings.findIndex(
      (recording) => recording.recording_id === selectedRecordingId,
    );
    if (selectedIndex === -1) return [main];

    // Drawn on top of the point itself so the selection stays visible in a
    // crowd without recolouring it (its colour is the score being read).
    const selected = {
      type,
      mode: "markers",
      name: "selected",
      x: [dimensions.x[selectedIndex]],
      y: [dimensions.y[selectedIndex]],
      ...(is3D ? { z: [(dimensions as { z: number[] }).z[selectedIndex]] } : {}),
      hoverinfo: "skip",
      showlegend: false,
      marker: {
        size: is3D ? 8 : 15,
        color: "rgba(0,0,0,0)",
        line: { width: 3, color: SELECTED_COLOR },
      },
    };
    return [main, selected];
  }, [recordings, coordinates, is3D, selectedRecordingId]);

  const layout = useMemo(() => {
    const axisTitle = (n: number) => ({ text: `component ${n}`, font: { size: 10 } });
    const base = {
      autosize: true,
      margin: { l: is3D ? 0 : 36, r: 0, t: 8, b: is3D ? 0 : 32 },
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      showlegend: false,
      hovermode: "closest",
      // Keeps the user's zoom/rotation across selection changes; a new
      // viewKey (different method or 2D/3D) resets it.
      uirevision: viewKey,
    };
    if (is3D) {
      const axis = (n: number) => ({ title: axisTitle(n), showticklabels: false });
      return { ...base, scene: { xaxis: axis(1), yaxis: axis(2), zaxis: axis(3) } };
    }
    // Reduced coordinates have no meaningful scale, so the ticks are noise.
    const axis = (n: number) => ({
      title: axisTitle(n),
      showticklabels: false,
      zeroline: false,
    });
    return { ...base, xaxis: axis(1), yaxis: axis(2) };
  }, [is3D, viewKey]);

  return (
    <Plot
      data={data as any}
      layout={layout as any}
      config={{ displaylogo: false, responsive: true }}
      useResizeHandler
      style={{ width: "100%", height: "100%" }}
      onClick={(event: any) => {
        const point = event.points?.find((candidate: any) => candidate.curveNumber === 0);
        const recordingId = point?.customdata?.[0];
        if (typeof recordingId === "string") onSelectRecording(recordingId);
      }}
    />
  );
};

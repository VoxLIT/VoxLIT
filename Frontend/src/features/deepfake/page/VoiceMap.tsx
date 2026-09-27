import { Suspense, lazy, useCallback, useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Box, Loader2, Orbit, RefreshCw, Sparkle, Square } from "lucide-react";
import type { DeepfakeEmbeddingProjection, RecordingInfo } from "../types";
import { usePalette } from "./theme";
import { PointPopup } from "./PointPopup";
import { Disclosure, ErrorNote, PrimaryButton } from "./ui";
import { VoiceMap2D, type MapPointerEvent } from "./VoiceMap2D";

const VoiceMap3D = lazy(() => import("./VoiceMap3D"));

export type ReductionMethod = "pca" | "umap" | "tsne";
const METHODS: { id: ReductionMethod; label: string; hint: string }[] = [
  { id: "pca", label: "PCA", hint: "linear projection, keeps the global structure" },
  { id: "umap", label: "UMAP", hint: "keeps neighbourhoods, good for clusters" },
  { id: "tsne", label: "t-SNE", hint: "pulls similar clips together, distances are not preserved" },
];

const POPUP_WIDTH = 270;
const POPUP_HEIGHT = 250;

interface VoiceMapProps {
  modelLabel: string;
  datasetSize: number;
  started: boolean;
  onStart: () => void;
  projection: DeepfakeEmbeddingProjection | null;
  loading: boolean;
  error: string | null;
  method: ReductionMethod;
  onMethodChange: (method: ReductionMethod) => void;
  is3D: boolean;
  on3DChange: (is3D: boolean) => void;
  onRefresh: () => void;
  recordingsById: Map<string, RecordingInfo>;
  selectedId: string;
  onSelect: (recordingId: string) => void;
}

/**
 * The centre of the page: every recording as a point in the space the
 * detector reads, coloured by its own score. Hover a point for its card,
 * click it to study it on either side of the map.
 */
export const VoiceMap = ({
  modelLabel,
  datasetSize,
  started,
  onStart,
  projection,
  loading,
  error,
  method,
  onMethodChange,
  is3D,
  on3DChange,
  onRefresh,
  recordingsById,
  selectedId,
  onSelect,
}: VoiceMapProps) => {
  const { REAL, MID, FAKE } = usePalette();
  const frame = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const [spread, setSpread] = useState(true);
  const closeTimer = useRef<number | undefined>(undefined);

  const cancelClose = () => window.clearTimeout(closeTimer.current);
  const scheduleClose = useCallback(() => {
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setHover(null), 220);
  }, []);
  useEffect(() => () => window.clearTimeout(closeTimer.current), []);
  // A new projection (or 2D ↔ 3D) unmounts the hovered point without a
  // mouse-leave, which would strand its card on screen.
  useEffect(() => setHover(null), [projection]);

  const handleHover = useCallback(
    (event: MapPointerEvent | null) => {
      if (!event) {
        scheduleClose();
        return;
      }
      const box = frame.current?.getBoundingClientRect();
      if (!box) return;
      cancelClose();
      setHover({ id: event.recordingId, x: event.clientX - box.left, y: event.clientY - box.top });
    },
    [scheduleClose],
  );

  const point = hover ? projection?.recordings.find((recording) => recording.recording_id === hover.id) : undefined;

  let popup = null;
  if (hover && point && frame.current) {
    const width = frame.current.clientWidth;
    const height = frame.current.clientHeight;
    const openRight = hover.x + 22 + POPUP_WIDTH < width;
    const left = openRight ? hover.x + 22 : Math.max(8, hover.x - 22 - POPUP_WIDTH);
    const top = Math.min(Math.max(8, hover.y - POPUP_HEIGHT / 2), Math.max(8, height - POPUP_HEIGHT - 8));
    popup = (
      <PointPopup
        key={point.recording_id}
        point={point}
        info={recordingsById.get(point.recording_id)}
        threshold={projection!.threshold}
        selected={point.recording_id === selectedId}
        left={left}
        top={top}
        originX={openRight ? "left" : "right"}
        onSelect={() => onSelect(point.recording_id)}
        onEnter={cancelClose}
        onLeave={scheduleClose}
      />
    );
  }

  const mapProps = projection
    ? {
        recordings: projection.recordings,
        coordinates: projection.coordinates,
        selectedId,
        hoveredId: hover?.id ?? null,
        onHover: handleHover,
        onSelect,
        spread,
      }
    : null;

  return (
    <div className="flex h-full flex-col">
      {/* panel header + controls */}
      <div className="df-panel-head flex flex-wrap items-center gap-2 px-3 py-2">
        <div className="flex items-center gap-1.5">
          <Orbit className="h-4 w-4 text-violet-300" />
          <h3 className="text-sm font-bold text-white">Voice map</h3>
        </div>
        <div className="ml-auto flex flex-wrap items-center gap-2">
          <div className="flex rounded-full bg-white/5 p-1 ring-1 ring-white/10" role="radiogroup" aria-label="Projection method">
            {METHODS.map((option) => (
              <button
                key={option.id}
                type="button"
                role="radio"
                aria-checked={method === option.id}
                title={option.hint}
                onClick={() => onMethodChange(option.id)}
                className="relative rounded-full px-3 py-1 text-xs font-semibold text-slate-300 transition-colors hover:text-white aria-checked:text-slate-950"
              >
                {method === option.id && (
                  <motion.span layoutId="df-method-pill" className="df-pill absolute inset-0 rounded-full" transition={{ type: "spring", stiffness: 400, damping: 30 }} />
                )}
                <span className="relative">{option.label}</span>
              </button>
            ))}
          </div>
          <div className="flex rounded-full bg-white/5 p-1 ring-1 ring-white/10" role="radiogroup" aria-label="Dimensions">
            {[false, true].map((three) => (
              <button
                key={String(three)}
                type="button"
                role="radio"
                aria-checked={is3D === three}
                onClick={() => on3DChange(three)}
                className="relative flex items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold text-slate-300 transition-colors hover:text-white aria-checked:text-slate-950"
              >
                {is3D === three && (
                  <motion.span layoutId="df-dim-pill" className="absolute inset-0 rounded-full df-pill" transition={{ type: "spring", stiffness: 400, damping: 30 }} />
                )}
                <span className="relative flex items-center gap-1">
                  {three ? <Box className="h-3 w-3" /> : <Square className="h-3 w-3" />}
                  {three ? "3D" : "2D"}
                </span>
              </button>
            ))}
          </div>
          <button
            type="button"
            aria-pressed={spread}
            onClick={() => setSpread((value) => !value)}
            title="Nudge overlapping points apart so every clip is visible. Off shows the raw projection."
            className={`flex items-center gap-1 rounded-full px-3 py-1.5 text-xs font-semibold ring-1 ring-white/10 transition ${spread ? "df-pill text-slate-950" : "text-slate-300 hover:bg-white/10 hover:text-white"}`}
          >
            <Sparkle className="h-3 w-3" /> Spread overlaps
          </button>
          <button
            type="button"
            onClick={onRefresh}
            disabled={!started || loading}
            aria-label="Refresh the voice map"
            className="rounded-full p-2 text-slate-300 ring-1 ring-white/10 transition hover:bg-white/10 hover:text-white disabled:opacity-40"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? "animate-spin" : ""}`} />
          </button>
        </div>
      </div>

      <div className="flex flex-1 flex-col gap-2 p-3">
      {/* the map frame */}
      <div
        ref={frame}
        className="relative h-[460px] overflow-hidden rounded-2xl sm:h-[560px] df-well ring-1 ring-white/10"
        onMouseLeave={scheduleClose}
      >
        {!started && (
          <div className="absolute inset-0 grid place-items-center p-6">
            <div className="max-w-md text-center">
              {/* a miniature of the real thing, so the button's outcome is visible before pressing it */}
              <MapPreview />
              <div className="mt-3 text-sm font-bold text-white">One point per clip</div>
              <p className="mx-auto mt-1 max-w-sm text-xs text-slate-400">
                Plot all {datasetSize || ""} clips by how {modelLabel} hears them. Clips it hears alike sit
                together; the colour is the detector&apos;s own score, never the dataset&apos;s answer.
              </p>
              <PrimaryButton onClick={onStart} disabled={datasetSize === 0} className="mt-3">
                <Orbit className="h-3.5 w-3.5" /> {datasetSize ? "Map the voices" : "Waiting for the clip list"}
              </PrimaryButton>
            </div>
          </div>
        )}

        {started && !projection && loading && (
          <div className="absolute inset-0 grid place-items-center p-6 text-center">
            <div>
              <div className="relative mx-auto h-20 w-20">
                {[0, 1, 2].map((ring) => (
                  <motion.span
                    key={ring}
                    className="absolute inset-0 rounded-full border-2"
                    style={{ borderColor: [REAL, MID, FAKE][ring] }}
                    animate={{ scale: [0.4, 1.4], opacity: [1, 0] }}
                    transition={{ duration: 2, repeat: Infinity, delay: ring * 0.6, ease: "easeOut" }}
                  />
                ))}
              </div>
              <p className="mt-4 max-w-xs text-sm text-slate-300">
                Listening to every clip. The first map for a model takes a few minutes (it downloads the
                checkpoint); after that it is quick.
              </p>
            </div>
          </div>
        )}

        {mapProps && projection!.n_components !== 3 && (
          <div className="absolute inset-0">
            <VoiceMap2D {...mapProps} />
          </div>
        )}
        {mapProps && projection!.n_components === 3 && (
          <Suspense
            fallback={
              <div className="absolute inset-0 grid place-items-center">
                <Loader2 className="h-6 w-6 animate-spin text-violet-300" />
              </div>
            }
          >
            <div className="absolute inset-0">
              <VoiceMap3D {...mapProps} />
            </div>
          </Suspense>
        )}
        {projection && loading && (
          <div className="absolute left-3 top-3 flex items-center gap-2 df-overlay-chip rounded-full px-3 py-1 text-xs text-slate-200 backdrop-blur">
            <Loader2 className="h-3 w-3 animate-spin" /> Redrawing
          </div>
        )}

        <AnimatePresence>{popup}</AnimatePresence>
      </div>

      {error && <ErrorNote>{error}</ErrorNote>}

      {/* legend */}
      <div className="flex items-center gap-3 text-xs text-slate-300">
        <span style={{ color: REAL }} className="font-semibold">
          sounds real
        </span>
        <div className="relative h-2 flex-1 rounded-full" style={{ background: `linear-gradient(90deg, ${REAL}, ${MID}, ${FAKE})` }}>
          {projection && (
            <div
              className="df-marker absolute -top-1 h-4 w-0.5 rounded"
              style={{ left: `${projection.threshold * 100}%` }}
              title={`threshold ${projection.threshold.toFixed(2)}`}
            />
          )}
        </div>
        <span style={{ color: FAKE }} className="font-semibold">
          sounds synthetic
        </span>
      </div>
      <p className="text-xs text-slate-400">
        Hover a point to peek, click it to study it. The selected clip grows and pulses; dashed lines join it to
        its five nearest points on this map. {is3D ? "Drag to turn the map, scroll to move closer" : "Scroll to zoom into a cluster, drag to pan, double-click to reset"}.
        {spread && " Overlapping points are nudged apart just enough to see each one; turn off Spread overlaps for the raw projection."}
        {projection?.recordings.some((recording) => recording.uploaded) &&
          " Points ringed and labelled \"you\" are your own clips, placed among the dataset."}
      </p>

      {projection && (
        <Disclosure title="How this map is made">
          {projection.reduction_method_used !== projection.reduction_method && (
            <p className="text-amber-200">
              {projection.reduction_method.toUpperCase()} could not run on this data, showing{" "}
              {projection.reduction_method_used.toUpperCase()} instead.
            </p>
          )}
          <p>
            Each clip is the {projection.embedding_dimension}-dimensional vector {projection.model_label}&apos;s
            classification head reads, reduced to {projection.n_components}D with{" "}
            {METHODS.find((option) => option.id === projection.reduction_method_used)?.label ??
              projection.reduction_method_used}{" "}
            ({projection.total_recordings} clips). Colour is the detector&apos;s own spoof score, never the
            dataset&apos;s label.
          </p>
          <p className="text-xs text-slate-400">
            The projection is for looking, not measuring: distances on screen are not distances inside the model,
            least of all with t-SNE and UMAP. The axes have no units, so they are not drawn. Threshold{" "}
            {projection.threshold.toFixed(2)}
            {projection.threshold_calibrated ? "" : " (uncalibrated)"}.
          </p>
        </Disclosure>
      )}
      </div>
    </div>
  );
};

/**
 * A small, honest preview of the map: two loose groups of points on the same
 * blue→red score ramp the real view uses, drifting gently. It shows what the
 * button produces without pretending to be data.
 */
const MapPreview = () => {
  const { scoreColor, ink } = usePalette();
  // Fixed layout: a genuine-leaning group on the left, a synthetic-leaning
  // one on the right, and a few clips in between.
  const dots = [
    [18, 46, 0.05], [26, 58, 0.08], [32, 38, 0.06], [40, 52, 0.1], [30, 68, 0.04],
    [46, 44, 0.12], [22, 34, 0.07], [38, 62, 0.09],
    [92, 50, 0.45], [104, 40, 0.52], [86, 62, 0.38],
    [140, 34, 0.88], [150, 48, 0.92], [160, 40, 0.85], [168, 56, 0.94], [148, 64, 0.9],
    [176, 44, 0.96], [158, 30, 0.82], [134, 54, 0.9], [184, 52, 0.93],
  ] as const;

  return (
    <svg viewBox="0 0 200 96" className="mx-auto h-24 w-full max-w-xs" role="img" aria-label="Preview: clips plotted as points, blue where the detector hears a real voice and red where it hears a synthetic one">
      <rect x={0.5} y={0.5} width={199} height={95} rx={3} fill="none" stroke={ink(0.08)} />
      {dots.map(([cx, cy, score], index) => (
        <circle
          key={index}
          cx={cx}
          cy={cy}
          r={index % 5 === 0 ? 4 : 3}
          fill={scoreColor(score)}
          opacity={0.85}
          className="df-float"
          style={{ animationDelay: `${(index % 7) * 0.4}s` }}
        />
      ))}
    </svg>
  );
};

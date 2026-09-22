import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, Box, Check, Loader2, MousePointerClick, Orbit, Play, Square } from "lucide-react";
import type { DiarizationResult, DiarizationSegment, ProjectionResult } from "../types";
import { nearestByCosine } from "./neighbours";
import { SegmentMap2D, type MapPointerEvent } from "./SegmentMap2D";
import { POPUP_HEIGHT, POPUP_WIDTH, SegmentPopup } from "./SegmentPopup";
import { confidenceWords, formatClock } from "./segmentWords";
import { usePalette } from "./theme";
import { Disclosure, Finding } from "./ui";

const NEIGHBOURS = 5;

interface SegmentMapProps {
  result: DiarizationResult;
  projection: ProjectionResult | null;
  projectionError: string | null;
  /** True while /run or /projection is still in flight. */
  isRunning: boolean;
  embeddingDimension: number | null;
  selectedId: string | null;
  /** Shared with the timeline: hovering a dot highlights its segment there. */
  hoveredId: string | null;
  onHover: (segmentId: string | null) => void;
  /** Select a segment and play it from its start. */
  onSelect: (segmentId: string) => void;
  /** Play one segment without changing the selection. */
  onPlay: (segmentId: string) => void;
}

/**
 * Step 3: which voices sound alike. The selected segment on the left, the
 * map of every segment in the centre, and the selected segment's nearest
 * segments on the right. The side columns come from `result.embeddings`, so
 * they keep working when the 2D projection could not be computed.
 */
export const SegmentMap = ({
  result,
  projection,
  projectionError,
  isRunning,
  embeddingDimension,
  selectedId,
  hoveredId,
  onHover,
  onSelect,
  onPlay,
}: SegmentMapProps) => {
  const { speakerColor, WARN } = usePalette();
  const colourOf = (speaker: string) => speakerColor(result.speakers, speaker);
  const segmentsById = useMemo(() => new Map(result.segments.map((s) => [s.id, s])), [result.segments]);
  const selected = selectedId !== null ? segmentsById.get(selectedId) : undefined;
  const neighbours = useMemo(
    () => (selectedId !== null ? nearestByCosine(result.embeddings, selectedId, NEIGHBOURS) : []),
    [result.embeddings, selectedId],
  );
  const notEmbedded = result.segments.length - Object.keys(result.embeddings).length;

  // --- hover card ------------------------------------------------------------
  const frame = useRef<HTMLDivElement>(null);
  const [hover, setHover] = useState<{ id: string; x: number; y: number } | null>(null);
  const closeTimer = useRef<number | undefined>(undefined);
  const cancelClose = () => window.clearTimeout(closeTimer.current);
  const scheduleClose = useCallback(() => {
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => {
      setHover(null);
      onHover(null);
    }, 220);
  }, [onHover]);
  useEffect(() => () => window.clearTimeout(closeTimer.current), []);
  // A new projection unmounts the hovered point without a mouse-leave, which
  // would strand its card on screen.
  useEffect(() => setHover(null), [projection]);

  const handleHover = useCallback(
    (event: MapPointerEvent | null) => {
      if (!event) {
        scheduleClose();
        return;
      }
      const box = frame.current?.getBoundingClientRect();
      if (!box) return;
      window.clearTimeout(closeTimer.current);
      setHover({ id: event.segmentId, x: event.clientX - box.left, y: event.clientY - box.top });
      onHover(event.segmentId);
    },
    [scheduleClose, onHover],
  );

  const hoveredSegment = hover ? segmentsById.get(hover.id) : undefined;
  let popup = null;
  if (hover && hoveredSegment && frame.current) {
    const width = frame.current.clientWidth;
    const height = frame.current.clientHeight;
    const openRight = hover.x + 22 + POPUP_WIDTH < width;
    const left = openRight ? hover.x + 22 : Math.max(8, hover.x - 22 - POPUP_WIDTH);
    const top = Math.min(Math.max(8, hover.y - POPUP_HEIGHT / 2), Math.max(8, height - POPUP_HEIGHT - 8));
    popup = (
      <SegmentPopup
        key={hoveredSegment.id}
        segment={hoveredSegment}
        colour={colourOf(hoveredSegment.speaker)}
        selected={hoveredSegment.id === selectedId}
        left={left}
        top={top}
        originX={openRight ? "left" : "right"}
        onPlay={() => onPlay(hoveredSegment.id)}
        onSelect={() => onSelect(hoveredSegment.id)}
        onEnter={cancelClose}
        onLeave={scheduleClose}
      />
    );
  }

  // --- finding -------------------------------------------------------------
  let finding = null;
  if (selected && neighbours.length > 0) {
    const others = neighbours.filter((n) => segmentsById.get(n.id)?.speaker !== selected.speaker);
    const closest = segmentsById.get(neighbours[0].id);
    if (closest && closest.speaker !== selected.speaker) {
      finding = (
        <Finding
          alarming
          title={`Its closest match belongs to ${closest.speaker}.`}
          detail={`This moment of ${selected.speaker} sounds most like ${closest.speaker} to the model, so these two voices are the easiest to mix up here.`}
        />
      );
    } else if (others.length > 0) {
      finding = (
        <Finding
          alarming
          title={`${others.length} of its ${neighbours.length} closest matches are other speakers.`}
          detail={`Its best match is ${selected.speaker} as expected, but other voices are not far behind.`}
        />
      );
    } else {
      finding = (
        <Finding
          title={`All ${neighbours.length} closest matches are ${selected.speaker} too.`}
          detail="To the model, this voice is well separated from the others."
        />
      );
    }
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <div className="flex items-center gap-2">
          <Orbit className="h-5 w-5 dz-accent-text" />
          <h3 className="font-display text-lg font-semibold text-white">Segment map</h3>
        </div>
        <div className="ml-auto flex rounded-full bg-white/5 p-1 ring-1 ring-white/10" role="radiogroup" aria-label="Dimensions">
          <button
            type="button"
            role="radio"
            aria-checked
            className="relative flex items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold text-slate-950"
          >
            <span className="absolute inset-0 rounded-full dz-pill" />
            <span className="relative flex items-center gap-1">
              <Square className="h-3 w-3" /> 2D
            </span>
          </button>
          {/* The projection endpoint only returns 2D PCA today. aria-disabled, not
              `disabled`, so the "coming soon" tooltip still shows on hover. */}
          <button
            type="button"
            role="radio"
            aria-checked={false}
            aria-disabled="true"
            title="coming soon"
            className="flex cursor-not-allowed items-center gap-1 rounded-full px-3 py-1 text-xs font-semibold text-slate-400 opacity-60"
          >
            <Box className="h-3 w-3" /> 3D
            <span className="rounded-full bg-white/10 px-1.5 text-[10px] font-medium">soon</span>
          </button>
        </div>
      </div>

      <div className="grid gap-4 lg:grid-cols-[260px_minmax(0,1fr)_300px]">
        {/* left: the selected segment */}
        <aside aria-label="Selected segment" className="dz-card rounded-2xl p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Selected</div>
          <AnimatePresence mode="wait">
            {selected ? (
              <motion.div
                key={selected.id}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -8 }}
                className="mt-3 space-y-3"
              >
                <SpeakerChip speaker={selected.speaker} colour={colourOf(selected.speaker)} />
                <div className="font-mono text-lg text-white">
                  {formatClock(selected.start)} – {formatClock(selected.end)}
                </div>
                <dl className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <dt className="text-slate-500">length</dt>
                    <dd className="font-mono text-slate-200">{(selected.end - selected.start).toFixed(1)} s</dd>
                  </div>
                  <div>
                    <dt className="text-slate-500">the model was</dt>
                    <dd className="text-slate-200">{confidenceWords(selected.confidence_bucket)}</dd>
                  </div>
                </dl>
                <button
                  type="button"
                  onClick={() => onPlay(selected.id)}
                  className="flex w-full items-center justify-center gap-1.5 rounded-lg bg-white/10 px-3 py-2 text-xs font-semibold text-white transition hover:bg-white/20"
                >
                  <Play className="h-3.5 w-3.5" /> Play this segment
                </button>
              </motion.div>
            ) : (
              <motion.p key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="mt-3 flex items-start gap-2 text-sm text-slate-400">
                <MousePointerClick className="mt-0.5 h-4 w-4 shrink-0" />
                Click a dot or a timeline segment.
              </motion.p>
            )}
          </AnimatePresence>
        </aside>

        {/* centre: the map */}
        <div
          ref={frame}
          className="dz-well relative h-[380px] overflow-hidden rounded-2xl ring-1 ring-white/10 sm:h-[480px]"
          onMouseLeave={scheduleClose}
        >
          {projection ? (
            <SegmentMap2D
              points={projection.points}
              segmentsById={segmentsById}
              speakers={result.speakers}
              selectedId={selectedId}
              hoveredId={hoveredId}
              neighbourIds={neighbours.map((n) => n.id)}
              onHover={handleHover}
              onSelect={onSelect}
            />
          ) : projectionError ? (
            <div role="status" className="absolute inset-0 grid place-items-center p-6">
              <div className="max-w-sm text-center">
                <AlertTriangle className="mx-auto h-6 w-6" style={{ color: WARN }} />
                <p className="mt-2 text-sm text-slate-200">Couldn&apos;t lay out the map: {projectionError}</p>
                <p className="mt-1 text-xs text-slate-400">The timeline and the nearest-segment list still work.</p>
              </div>
            </div>
          ) : isRunning ? (
            <div role="status" className="absolute inset-0 grid place-items-center text-sm text-slate-400">
              <span className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" /> Laying out the map…
              </span>
            </div>
          ) : null}
          <AnimatePresence>{popup}</AnimatePresence>
        </div>

        {/* right: nearest segments by cosine in the model's own space */}
        <aside aria-label="Nearest segments" className="dz-card rounded-2xl p-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">Sounds most like</div>
          {!selected ? (
            <p className="mt-3 text-sm text-slate-400">Select a segment to see the ones that sound most like it.</p>
          ) : neighbours.length === 0 ? (
            <p className="mt-3 text-sm text-slate-400">Too short to have a voice fingerprint (under 0.4 s).</p>
          ) : (
            <ol className="mt-3 space-y-2">
              {neighbours.map((neighbour, index) => {
                const segment = segmentsById.get(neighbour.id);
                if (!segment) return null;
                const same = segment.speaker === selected.speaker;
                return (
                  <motion.li
                    key={`${selected.id}-${neighbour.id}`}
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    data-testid="neighbour-row"
                    className="flex items-center gap-2 rounded-xl bg-white/[0.03] p-2 ring-1 ring-white/5"
                  >
                    <button
                      type="button"
                      onClick={() => onSelect(segment.id)}
                      aria-label={`Select ${segment.speaker} at ${formatClock(segment.start)}`}
                      className="min-w-0 flex-1 text-left"
                    >
                      <SpeakerChip speaker={segment.speaker} colour={colourOf(segment.speaker)} />
                      <div className="mt-1 font-mono text-xs text-slate-300">{formatClock(segment.start)}</div>
                      <div className="mt-0.5 flex items-center gap-1 text-xs" style={same ? undefined : { color: WARN }}>
                        {same ? (
                          <span className="flex items-center gap-1 text-slate-400">
                            same speaker <Check className="h-3 w-3" />
                          </span>
                        ) : (
                          <>
                            different speaker <AlertTriangle className="h-3 w-3" />
                          </>
                        )}
                      </div>
                    </button>
                    <button
                      type="button"
                      onClick={() => onPlay(segment.id)}
                      aria-label={`Play ${segment.speaker} at ${formatClock(segment.start)}`}
                      className="rounded-full bg-white/10 p-2 text-white transition hover:bg-white/20"
                    >
                      <Play className="h-3.5 w-3.5" />
                    </button>
                  </motion.li>
                );
              })}
            </ol>
          )}
        </aside>
      </div>

      {finding}

      <Disclosure>
        <p>
          Each dot is one segment&apos;s speaker embedding
          {embeddingDimension ? ` (${embeddingDimension} numbers)` : ""}, projected to 2D with PCA. Distances on the
          map are approximate. The &ldquo;sounds most like&rdquo; list uses cosine similarity in the full embedding
          space, so it can disagree with what looks close on the map.
        </p>
        <p>
          {Object.keys(result.embeddings).length} of {result.segments.length} segments are on the map.
          {notEmbedded > 0 && ` ${notEmbedded} were too short to embed (under 0.4 s) and have no dot.`}
        </p>
        {selected && neighbours.length > 0 && (
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-500">
                <th className="font-normal">segment</th>
                <th className="font-normal">speaker</th>
                <th className="text-right font-normal">cosine similarity</th>
              </tr>
            </thead>
            <tbody className="font-mono">
              {neighbours.map((neighbour) => (
                <tr key={neighbour.id}>
                  <td>{neighbour.id}</td>
                  <td>{segmentsById.get(neighbour.id)?.speaker}</td>
                  <td className="text-right">{neighbour.similarity.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Disclosure>
    </div>
  );
};

const SpeakerChip = ({ speaker, colour }: { speaker: DiarizationSegment["speaker"]; colour: string }) => (
  <span
    className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-xs font-semibold text-white"
    style={{ background: `color-mix(in srgb, ${colour} 22%, transparent)`, boxShadow: `inset 0 0 0 1px ${colour}` }}
  >
    <span className="h-2 w-2 rounded-full" style={{ background: colour }} />
    {speaker}
  </span>
);

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Grid3x3, Play, SkipForward } from "lucide-react";
import type { DiarizationResult, DiarizationSegment } from "../types";
import { formatClock } from "./segmentWords";
import {
  mostAlikeSpeakers,
  orderSegments,
  similarityMatrix,
  similarityWords,
  SOMEWHAT_ALIKE,
  uncertainSegments,
  VERY_ALIKE,
} from "./similarity";
import { usePalette } from "./theme";
import { Disclosure, Finding, SpeakerChip } from "./ui";

const DEFAULT_SIZE = 480;
const MAX_SIZE = 560;
const MIN_SIZE = 240;
const STRIP = 8; // speaker colour strip along both axes
const UNCERTAIN_PREVIEW = 8;
const POPUP_WIDTH = 260;
const POPUP_HEIGHT = 190;

interface SimilarityViewProps {
  result: DiarizationResult;
  selectedId: string | null;
  /** Select a segment and play it from its start. */
  onSelect: (segmentId: string) => void;
  /** Play one segment without changing the selection. */
  onPlay: (segmentId: string) => void;
  /** Play segment A, then segment B. */
  onPlayPair: (aId: string, bId: string) => void;
}

const hexToRgb = (hex: string): [number, number, number] => {
  const value = parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
};

interface Hover {
  row: number;
  col: number;
  x: number;
  y: number;
}

/**
 * Step 4: where the model hesitated. Every embedded segment against every
 * other, on a canvas so it stays fast for a full meeting. Clicking a cell plays
 * the two segments back to back; hovering one explains the pair in words.
 */
export const SimilarityView = ({ result, selectedId, onSelect, onPlay, onPlayPair }: SimilarityViewProps) => {
  const { speakerColor, CANVAS, HEAT } = usePalette();
  const colourOf = (speaker: string) => speakerColor(result.speakers, speaker);
  const [groupBySpeaker, setGroupBySpeaker] = useState(false);
  const [showAllUncertain, setShowAllUncertain] = useState(false);

  const ordered = useMemo(
    () => orderSegments(result.segments, result.embeddings, result.speakers, groupBySpeaker),
    [result.segments, result.embeddings, result.speakers, groupBySpeaker],
  );
  const n = ordered.length;
  const matrix = useMemo(() => similarityMatrix(ordered, result.embeddings), [ordered, result.embeddings]);
  // Independent of the ordering: a mean over every cross-speaker pair.
  const pair = useMemo(() => mostAlikeSpeakers(ordered, matrix, result.speakers), [ordered, matrix, result.speakers]);
  const uncertain = useMemo(() => uncertainSegments(result.segments), [result.segments]);
  const unscored = result.segments.filter((s) => s.confidence === null).length;

  // --- size: square, as wide as the column allows --------------------------
  const holder = useRef<HTMLDivElement>(null);
  const [size, setSize] = useState(DEFAULT_SIZE);
  useEffect(() => {
    const element = holder.current;
    if (!element) return;
    const measure = () => {
      const width = element.clientWidth;
      // jsdom and a not-yet-laid-out column report 0: keep the default.
      if (width > 0) setSize(Math.max(MIN_SIZE, Math.min(MAX_SIZE, Math.floor(width))));
    };
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  // --- draw ------------------------------------------------------------------
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || n === 0) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(dpr, 0, 0, dpr, 0, 0);
    context.clearRect(0, 0, size, size);

    const low = hexToRgb(CANVAS);
    const high = hexToRgb(HEAT);
    const cellColor = (value: number) => {
      const t = Math.max(0, Math.min(1, value));
      const channel = (i: number) => Math.round(low[i] + (high[i] - low[i]) * t);
      return `rgb(${channel(0)}, ${channel(1)}, ${channel(2)})`;
    };

    const inner = size - STRIP;
    const cell = inner / n;
    for (let i = 0; i < n; i++) {
      // Speaker strips: top (columns) and left (rows)
      context.fillStyle = colourOf(ordered[i].speaker);
      context.fillRect(STRIP + i * cell, 0, Math.ceil(cell), STRIP - 1);
      context.fillRect(0, STRIP + i * cell, STRIP - 1, Math.ceil(cell));
      for (let j = 0; j < n; j++) {
        context.fillStyle = cellColor(matrix[i * n + j]);
        context.fillRect(STRIP + j * cell, STRIP + i * cell, Math.ceil(cell), Math.ceil(cell));
      }
    }
    // colourOf is derived from speakerColor + speakers, both listed.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [matrix, ordered, n, size, CANVAS, HEAT, speakerColor, result.speakers]);

  const cellFromEvent = (event: React.MouseEvent<HTMLCanvasElement>): { row: number; col: number } | null => {
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left - STRIP;
    const y = event.clientY - rect.top - STRIP;
    const inner = size - STRIP;
    if (x < 0 || y < 0 || x >= inner || y >= inner || n === 0) return null;
    return {
      row: Math.min(n - 1, Math.floor((y / inner) * n)),
      col: Math.min(n - 1, Math.floor((x / inner) * n)),
    };
  };

  // --- hover card: stays open while the pointer is on it, so ▶▶ is clickable --
  const [hover, setHover] = useState<Hover | null>(null);
  const closeTimer = useRef<number | undefined>(undefined);
  const cancelClose = () => window.clearTimeout(closeTimer.current);
  const scheduleClose = useCallback(() => {
    window.clearTimeout(closeTimer.current);
    closeTimer.current = window.setTimeout(() => setHover(null), 220);
  }, []);
  useEffect(() => () => window.clearTimeout(closeTimer.current), []);
  // A re-ordering moves every cell out from under the card.
  useEffect(() => setHover(null), [ordered]);

  const a = hover ? ordered[hover.row] : undefined;
  const b = hover ? ordered[hover.col] : undefined;
  let popup = null;
  if (hover && a && b) {
    const similarity = matrix[hover.row * n + hover.col];
    const openRight = hover.x + 18 + POPUP_WIDTH < size;
    const left = openRight ? hover.x + 18 : Math.max(4, hover.x - 18 - POPUP_WIDTH);
    const top = Math.min(Math.max(4, hover.y - POPUP_HEIGHT / 2), Math.max(4, size - POPUP_HEIGHT - 4));
    const same = a.id === b.id;
    popup = (
      <motion.div
        key="pair"
        role="dialog"
        aria-label="Compare two segments"
        initial={{ opacity: 0, scale: 0.6, filter: "blur(10px)", y: 10 }}
        animate={{ opacity: 1, scale: 1, filter: "blur(0px)", y: 0, left, top }}
        exit={{ opacity: 0, scale: 0.85, filter: "blur(6px)", transition: { duration: 0.15 } }}
        transition={{ type: "spring", stiffness: 420, damping: 30 }}
        style={{ width: POPUP_WIDTH, transformOrigin: `${openRight ? "0%" : "100%"} 50%` }}
        onMouseEnter={cancelClose}
        onMouseLeave={scheduleClose}
        className="dz-popup-border absolute z-30 rounded-2xl"
      >
        <div className="dz-panel space-y-2 rounded-2xl p-4 shadow-2xl backdrop-blur-xl">
          <PairRow segment={a} colour={colourOf(a.speaker)} />
          {!same && <PairRow segment={b} colour={colourOf(b.speaker)} />}
          <p className="text-sm text-slate-200">
            {same ? "The same moment, compared with itself." : <>These two moments {similarityWords(similarity)}.</>}
          </p>
          <button
            type="button"
            onClick={() => onPlayPair(a.id, b.id)}
            className="flex w-full items-center justify-center gap-1.5 rounded-lg dz-primary-btn px-3 py-2 text-xs font-semibold transition hover:brightness-110"
          >
            {same ? <Play className="h-3.5 w-3.5" /> : <SkipForward className="h-3.5 w-3.5" />}
            {same ? "Play it" : "Play both"}
          </button>
        </div>
      </motion.div>
    );
  }

  // --- finding -----------------------------------------------------------------
  let finding;
  if (pair) {
    finding = (
      <Finding
        alarming={pair.similarity >= SOMEWHAT_ALIKE}
        title={`${pair.a} and ${pair.b} sound the most alike, so these are the voices the system is most likely to mix up.`}
        detail={`On average, their moments ${similarityWords(pair.similarity)}.`}
      />
    );
  } else if (result.speakers.length < 2) {
    finding = <Finding title="Only one voice was found, so there is nothing to mix up." />;
  } else {
    finding = <Finding title="Too few moments were long enough to compare the voices." />;
  }

  const shownUncertain = showAllUncertain ? uncertain : uncertain.slice(0, UNCERTAIN_PREVIEW);
  const wipeKey = `${result.model}|${result.recording_id}|${groupBySpeaker}`;

  return (
    <div className="space-y-4">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_360px]">
        <div ref={holder} className="dz-card min-w-0 rounded-2xl p-4">
          <div className="mb-3 flex items-center gap-2">
            <Grid3x3 className="h-5 w-5 dz-accent-text" />
            <h3 className="font-display text-lg font-semibold text-white">Every moment against every other</h3>
          </div>
          {n === 0 ? (
            <p className="text-sm text-slate-400">No segment was long enough to compare (all under 0.4 s).</p>
          ) : (
            <>
              <div className="relative mx-auto" style={{ width: size, height: size }} onMouseLeave={scheduleClose}>
                <canvas
                  key={wipeKey}
                  ref={canvasRef}
                  role="img"
                  aria-label={`Similarity between ${n} segments. Brighter cells sound more alike.`}
                  style={{ width: size, height: size }}
                  className="dz-wipe cursor-crosshair rounded-lg"
                  onMouseMove={(event) => {
                    const position = cellFromEvent(event);
                    if (!position) return scheduleClose();
                    cancelClose();
                    const rect = event.currentTarget.getBoundingClientRect();
                    setHover({ ...position, x: event.clientX - rect.left, y: event.clientY - rect.top });
                  }}
                  onClick={(event) => {
                    const position = cellFromEvent(event);
                    if (!position) return;
                    onPlayPair(ordered[position.row].id, ordered[position.col].id);
                  }}
                />
                <AnimatePresence>{popup}</AnimatePresence>
              </div>
              <p className="mt-3 text-center text-xs text-slate-400">
                Brighter = more alike. Hover a cell to compare two moments, click it to hear both.
              </p>
            </>
          )}
        </div>

        <div className="space-y-4">
          {finding}

          <aside aria-label="Uncertain segments" className="dz-card rounded-2xl p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Where it was unsure {uncertain.length > 0 && `· ${uncertain.length}`}
            </div>
            {uncertain.length === 0 ? (
              <p className="mt-3 text-sm text-slate-400">
                The model was at least fairly sure about every segment it could score.
              </p>
            ) : (
              <ol className="mt-3 space-y-2">
                {shownUncertain.map((segment, index) => (
                  <motion.li
                    key={segment.id}
                    initial={{ opacity: 0, x: 12 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: Math.min(index * 0.04, 0.4) }}
                    data-testid="uncertain-row"
                    className={`flex items-center gap-2 rounded-xl p-2 ring-1 ${
                      segment.id === selectedId ? "bg-white/10 ring-white/20" : "bg-white/[0.03] ring-white/5"
                    }`}
                  >
                    <button
                      type="button"
                      onClick={() => onSelect(segment.id)}
                      aria-label={`Select ${segment.speaker} at ${formatClock(segment.start)}`}
                      className="flex min-w-0 flex-1 items-center gap-2 text-left"
                    >
                      <SpeakerChip speaker={segment.speaker} colour={colourOf(segment.speaker)} />
                      <span className="font-mono text-xs text-slate-300">
                        {formatClock(segment.start)} – {formatClock(segment.end)}
                      </span>
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
                ))}
              </ol>
            )}
            {uncertain.length > UNCERTAIN_PREVIEW && (
              <button
                type="button"
                onClick={() => setShowAllUncertain((value) => !value)}
                className="mt-3 text-xs font-semibold dz-accent-text hover:underline"
              >
                {showAllUncertain ? "Show fewer" : `Show all ${uncertain.length}`}
              </button>
            )}
            {unscored > 0 && (
              <p className="mt-3 text-xs text-slate-500">
                {unscored} segment{unscored === 1 ? " was" : "s were"} too short to score (under 0.4 s) and {unscored === 1 ? "is" : "are"} not
                listed.
              </p>
            )}
          </aside>
        </div>
      </div>

      <Disclosure>
        <p>
          Each cell is the cosine similarity between two segments&apos; speaker embeddings, the same fingerprints the
          clustering groups. 1 means identical; clustering puts segments with high similarity under one speaker, so
          bright cells between <em>different</em> speakers mark where it had to choose.
        </p>
        <div>
          <div className="mb-1.5 text-xs text-slate-400">Order the rows and columns by</div>
          <div className="inline-flex rounded-full bg-white/5 p-1 ring-1 ring-white/10" role="radiogroup" aria-label="Matrix order">
            {[
              { grouped: false, label: "Time" },
              { grouped: true, label: "Speaker" },
            ].map((option) => {
              const active = option.grouped === groupBySpeaker;
              return (
                <button
                  key={option.label}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  onClick={() => setGroupBySpeaker(option.grouped)}
                  className="relative rounded-full px-3 py-1 text-xs font-semibold text-slate-300 transition-colors aria-checked:text-slate-950"
                >
                  {active && (
                    <motion.span
                      layoutId="dz-matrix-order"
                      className="absolute inset-0 rounded-full dz-pill"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                  <span className="relative">{option.label}</span>
                </button>
              );
            })}
          </div>
          <p className="mt-1.5 text-xs text-slate-400">
            Grouped by speaker, a well-separated run looks like bright squares along the diagonal.
          </p>
        </div>
        <div>
          <div className="h-2.5 w-full max-w-xs rounded-full" style={{ background: `linear-gradient(90deg, ${CANVAS}, ${HEAT})` }} />
          <div className="mt-1 flex max-w-xs justify-between font-mono text-[11px] text-slate-400">
            <span>≤ 0</span>
            <span>{SOMEWHAT_ALIKE}</span>
            <span>{VERY_ALIKE}</span>
            <span>1</span>
          </div>
          <p className="mt-1.5 text-xs text-slate-400">
            In words: ≥ {VERY_ALIKE} &ldquo;very alike&rdquo;, ≥ {SOMEWHAT_ALIKE} &ldquo;somewhat alike&rdquo;, below that
            &ldquo;different&rdquo;. These cut-offs are a rule of thumb, not calibrated thresholds.
          </p>
        </div>
        <p>
          {n} of {result.segments.length} segments are in the matrix.
          {result.segments.length - n > 0 && ` ${result.segments.length - n} were too short to embed (under 0.4 s).`}
          {pair && ` Mean similarity between ${pair.a} and ${pair.b}: ${pair.similarity.toFixed(3)}.`}
        </p>
      </Disclosure>
    </div>
  );
};

const PairRow = ({ segment, colour }: { segment: DiarizationSegment; colour: string }) => (
  <div className="flex items-center gap-2">
    <SpeakerChip speaker={segment.speaker} colour={colour} />
    <span className="font-mono text-xs text-slate-300">
      {formatClock(segment.start)} – {formatClock(segment.end)}
    </span>
  </div>
);

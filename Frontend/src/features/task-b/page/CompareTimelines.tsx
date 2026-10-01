import { motion } from "motion/react";
import type { DiarizationDelta, DiarizationSegment, PerturbationRunSummary } from "../types";
import { formatClock, segmentLabel } from "./segmentWords";
import { usePalette } from "./theme";
import { perturbedColourKey } from "./whatIfWords";

interface CompareTimelinesProps {
  original: PerturbationRunSummary;
  perturbed: PerturbationRunSummary;
  delta: DiarizationDelta;
  /** The original run's speaker order, so colours match Steps 2–4. */
  speakers: string[];
  selectedOriginalId: string | null;
  selectedPerturbedId: string | null;
  onSelectOriginal: (segmentId: string) => void;
  onSelectPerturbed: (segmentId: string) => void;
}

/**
 * Before and after on one time axis, with the spans where they disagree
 * pulsing between them. The perturbed run is coloured through the backend's
 * label alignment, so a voice keeps its colour across both timelines.
 */
export const CompareTimelines = ({
  original,
  perturbed,
  delta,
  speakers,
  selectedOriginalId,
  selectedPerturbedId,
  onSelectOriginal,
  onSelectPerturbed,
}: CompareTimelinesProps) => {
  const { speakerColor, WARN } = usePalette();
  const duration = original.duration || 1;
  const colours = perturbedColourKey(delta, speakers, perturbed.speakers);
  const originalColour = (label: string) => speakerColor(speakers, label);
  const perturbedColour = (label: string) => speakerColor(colours.speakers, colours.keyOf(label));

  // Perturbed lanes in the original's speaker order, unmatched voices last.
  const perturbedLanes = [...perturbed.speakers].sort(
    (a, b) => colours.speakers.indexOf(colours.keyOf(a)) - colours.speakers.indexOf(colours.keyOf(b)),
  );

  return (
    <div className="dz-card space-y-3 rounded-2xl p-4">
      <RunLanes
        title="Before"
        caption={runCaption(original)}
        lanes={speakers.filter((s) => original.speakers.includes(s))}
        laneName={(label) => label}
        segments={original.segments}
        duration={duration}
        colourOf={originalColour}
        selectedId={selectedOriginalId}
        onSelect={onSelectOriginal}
        testId="lane-before"
      />

      {/* What changed. Already aligned to the original's labels by the backend. */}
      <div className="flex items-center gap-3">
        <span className="w-28 shrink-0 text-xs font-semibold" style={{ color: WARN }}>
          changed
        </span>
        <div className="relative h-4 flex-1 overflow-hidden rounded bg-white/[0.04]" data-testid="diff-strip">
          {delta.diff_regions.map((region, index) => (
            <motion.div
              key={`${region.start}-${region.end}`}
              initial={{ opacity: 0, scaleY: 0 }}
              animate={{ opacity: 1, scaleY: 1 }}
              transition={{ delay: Math.min(index * 0.01, 0.6) }}
              className="dz-diff-pulse absolute top-0 h-full rounded-sm"
              style={{
                left: `${(region.start / duration) * 100}%`,
                width: `${Math.max(((region.end - region.start) / duration) * 100, 0.2)}%`,
                background: WARN,
              }}
              title={`The runs disagree ${formatClock(region.start)} – ${formatClock(region.end)}`}
            />
          ))}
        </div>
      </div>
      <p className="pl-[7.75rem] text-xs text-slate-400">
        {delta.diff_regions.length > 0
          ? "Highlighted where the two runs disagree about who is speaking."
          : "The two runs agree everywhere: the audio changed, the answer did not."}
      </p>

      <RunLanes
        title="After"
        caption={runCaption(perturbed)}
        lanes={perturbedLanes}
        laneName={(label) => delta.speaker_mapping[label] ?? `new: ${label}`}
        segments={perturbed.segments}
        duration={duration}
        colourOf={perturbedColour}
        selectedId={selectedPerturbedId}
        onSelect={onSelectPerturbed}
        testId="lane-after"
      />

      <div className="flex justify-between pl-[7.75rem] font-mono text-[11px] text-slate-500">
        <span>{formatClock(0)}</span>
        <span>{formatClock(duration)}</span>
      </div>
    </div>
  );
};

const runCaption = (run: PerturbationRunSummary) =>
  `${run.num_speakers} speaker${run.num_speakers === 1 ? "" : "s"}, ${run.segments.length} segment${run.segments.length === 1 ? "" : "s"}`;

const RunLanes = ({
  title,
  caption,
  lanes,
  laneName,
  segments,
  duration,
  colourOf,
  selectedId,
  onSelect,
  testId,
}: {
  title: string;
  caption: string;
  lanes: string[];
  laneName: (label: string) => string;
  segments: DiarizationSegment[];
  duration: number;
  colourOf: (label: string) => string;
  selectedId: string | null;
  onSelect: (segmentId: string) => void;
  testId: string;
}) => (
  <div data-testid={testId}>
    <div className="mb-1.5 flex items-baseline gap-2">
      <span className="font-display text-sm font-semibold text-white">{title}</span>
      <span className="text-xs text-slate-400">{caption}</span>
    </div>
    <div className="space-y-1">
      {lanes.map((lane) => {
        const colour = colourOf(lane);
        return (
          <div key={lane} className="flex items-center gap-3">
            <span className="flex w-28 shrink-0 items-center gap-1.5 truncate text-xs text-slate-300">
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: colour }} />
              <span className="truncate">{laneName(lane)}</span>
            </span>
            <div className="relative h-3.5 flex-1 rounded bg-white/[0.04]">
              {segments
                .filter((segment) => segment.speaker === lane)
                .map((segment, index) => {
                  const unsure = segment.confidence_bucket === "uncertain";
                  const selected = segment.id === selectedId;
                  return (
                    <motion.button
                      key={segment.id}
                      type="button"
                      aria-label={segmentLabel({ ...segment, speaker: laneName(lane) })}
                      aria-pressed={selected}
                      onClick={() => onSelect(segment.id)}
                      initial={{ scaleX: 0 }}
                      animate={{ scaleX: 1, scaleY: selected ? 1.6 : 1 }}
                      transition={{ duration: 0.4, delay: Math.min(index * 0.008, 0.6), ease: [0.2, 0.7, 0.3, 1] }}
                      className={`absolute top-0 h-full origin-left rounded-sm ${unsure ? "dz-hatch" : ""} ${
                        selected ? "dz-selected z-10" : ""
                      }`}
                      style={{
                        left: `${(segment.start / duration) * 100}%`,
                        width: `${Math.max(((segment.end - segment.start) / duration) * 100, 0.15)}%`,
                        // Uncertainty is hatching and opacity in the speaker's own colour, never a new colour.
                        background: unsure ? undefined : colour,
                        color: colour,
                        opacity: unsure ? 0.6 : segment.confidence_bucket === "medium" ? 0.8 : 1,
                      }}
                    />
                  );
                })}
            </div>
          </div>
        );
      })}
    </div>
  </div>
);

import { DiarizationTimeline } from "./DiarizationTimeline";
import { DiarizationDelta, PerturbationRunSummary } from "./types";

interface StackedTimelinesProps {
  original: PerturbationRunSummary;
  perturbed: PerturbationRunSummary;
  delta: DiarizationDelta;
  hoveredId: string | null;
  selectedId: string | null;
  onHover: (id: string | null) => void;
  onSelectOriginal: (segmentId: string) => void;
  onSelectPerturbed: (segmentId: string) => void;
}

/** The two runs share one time axis, so the strip can be positioned with the
 *  same percentage maths `DiarizationTimeline` uses for its bars. The
 *  `ml-[6.5rem]` matches that component's speaker-label gutter so the strip
 *  lines up with the lanes above and below it. */
export const StackedTimelines = ({
  original,
  perturbed,
  delta,
  hoveredId,
  selectedId,
  onHover,
  onSelectOriginal,
  onSelectPerturbed,
}: StackedTimelinesProps) => {
  const duration = original.duration || 1;

  return (
    <div className="space-y-3">
      <div>
        <div className="mb-1 flex items-center gap-2">
          <span className="text-xs font-medium">Original</span>
          <span className="text-xs text-muted-foreground">
            {original.num_speakers} speakers, {original.segments.length} segments
          </span>
        </div>
        <DiarizationTimeline
          segments={original.segments}
          speakers={original.speakers}
          duration={duration}
          hoveredId={hoveredId}
          selectedId={selectedId}
          onHover={onHover}
          onSelect={onSelectOriginal}
        />
      </div>

      {/* Difference strip. Regions come from the backend already aligned to
          the original's speaker labels, so this is a straight render — the
          client never recomputes the comparison. */}
      {/* The speaker palette's second colour is also red (types.ts), so a
          plain red bar here reads as another speaker lane. The gutter label
          and the diagonal hatch keep the strip distinguishable from the
          speaker bars directly above and below it. */}
      <div>
        <div className="flex items-center gap-2">
          <span className="w-24 shrink-0 truncate text-xs font-medium text-red-600">
            disagreement
          </span>
          <div className="relative h-4 flex-1 rounded bg-muted/40">
            {delta.diff_regions.map((region) => (
              <div
                key={`${region.start}-${region.end}`}
                className="absolute top-0 h-full rounded-sm border border-red-700/70"
                style={{
                  left: `${(region.start / duration) * 100}%`,
                  width: `${Math.max(((region.end - region.start) / duration) * 100, 0.2)}%`,
                  backgroundImage:
                    "repeating-linear-gradient(45deg, #dc2626, #dc2626 3px, #fca5a5 3px, #fca5a5 6px)",
                }}
                title={`Runs disagree ${region.start.toFixed(2)}s – ${region.end.toFixed(2)}s`}
              />
            ))}
          </div>
        </div>
        <div className="ml-[6.5rem] mt-1 text-[10px] text-muted-foreground">
          {delta.diff_regions.length > 0
            ? `${delta.diff_regions.length} region${
                delta.diff_regions.length === 1 ? "" : "s"
              } where the two runs disagree about who is speaking`
            : "The two runs agree everywhere — the perturbation changed the audio but not the diarization."}
        </div>
      </div>

      <div>
        <div className="mb-1 flex items-center gap-2">
          <span className="text-xs font-medium">Perturbed</span>
          <span className="text-xs text-muted-foreground">
            {perturbed.num_speakers} speakers, {perturbed.segments.length} segments
          </span>
        </div>
        <DiarizationTimeline
          segments={perturbed.segments}
          speakers={perturbed.speakers}
          duration={duration}
          hoveredId={hoveredId}
          selectedId={selectedId}
          onHover={onHover}
          onSelect={onSelectPerturbed}
        />
      </div>
    </div>
  );
};

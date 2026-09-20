import { Badge } from "@/components/ui/badge";
import { ArrowRight, Minus, Plus, Split, Merge } from "lucide-react";
import { DiarizationDelta, PerturbationSpec } from "./types";

interface DeltaSummaryCardProps {
  delta: DiarizationDelta;
  perturbation: PerturbationSpec;
  cached: boolean;
}

const Metric = ({
  label,
  value,
  hint,
}: {
  label: string;
  value: string;
  hint?: string;
}) => (
  <div className="rounded-lg border p-3">
    <div className="text-xs text-muted-foreground">{label}</div>
    <div className="mt-1 font-mono text-lg">{value}</div>
    {hint && <div className="mt-0.5 text-[10px] text-muted-foreground">{hint}</div>}
  </div>
);

export const DeltaSummaryCard = ({ delta, perturbation, cached }: DeltaSummaryCardProps) => {
  const { der_components: components } = delta;
  const structural =
    delta.appeared.length +
    delta.disappeared.length +
    delta.merged.length +
    delta.split.length;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary" className="font-mono text-xs">
          {perturbation.type}
        </Badge>
        {Object.entries(perturbation.params).map(([key, value]) => (
          <Badge key={key} variant="outline" className="font-mono text-[10px]">
            {key}={value}
          </Badge>
        ))}
        {cached && (
          <Badge variant="outline" className="text-[10px]">
            cached
          </Badge>
        )}
      </div>

      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Metric
          label="DER vs. original run"
          value={delta.der.toFixed(4)}
          // The single most important caveat in this whole feature.
          hint="Not accuracy — the original run is the reference, not ground truth"
        />
        <Metric
          label="Speakers"
          value={
            delta.num_speakers_delta === 0
              ? "unchanged"
              : `${delta.num_speakers_delta > 0 ? "+" : ""}${delta.num_speakers_delta}`
          }
        />
        <Metric
          label={`Boundary shifts > ${delta.boundary_shifts.threshold_seconds}s`}
          value={String(delta.boundary_shifts.count)}
        />
        <Metric
          label="Lost segments"
          value={String(delta.lost_segments.length)}
          hint="Had no counterpart in the perturbed run"
        />
      </div>

      <div className="rounded-lg border p-3">
        <div className="mb-2 text-xs text-muted-foreground">
          What drove the DER (seconds of reference speech)
        </div>
        <div className="grid grid-cols-2 gap-2 font-mono text-xs sm:grid-cols-4">
          <div>
            missed <span className="text-foreground">{components.missed_detection}</span>
          </div>
          <div>
            false alarm <span className="text-foreground">{components.false_alarm}</span>
          </div>
          <div>
            confusion <span className="text-foreground">{components.confusion}</span>
          </div>
          <div className="text-muted-foreground">
            total <span className="text-foreground">{components.total}</span>
          </div>
        </div>
      </div>

      <div className="rounded-lg border p-3">
        <div className="mb-2 text-xs text-muted-foreground">
          Speaker changes{" "}
          {structural === 0 && (
            <span className="text-foreground">— none, every speaker survived</span>
          )}
        </div>
        <div className="flex flex-wrap gap-1.5">
          {delta.appeared.map((speaker) => (
            <Badge key={`app-${speaker}`} variant="outline" className="text-[10px]">
              <Plus className="mr-1 h-3 w-3" /> {speaker} appeared
            </Badge>
          ))}
          {delta.disappeared.map((speaker) => (
            <Badge key={`dis-${speaker}`} variant="destructive" className="text-[10px]">
              <Minus className="mr-1 h-3 w-3" /> {speaker} disappeared
            </Badge>
          ))}
          {delta.split.map((item) => (
            <Badge key={`split-${item.original}`} variant="outline" className="text-[10px]">
              <Split className="mr-1 h-3 w-3" /> {item.original} → {item.into.join(" + ")}
            </Badge>
          ))}
          {delta.merged.map((item) => (
            <Badge key={`merge-${item.perturbed}`} variant="outline" className="text-[10px]">
              <Merge className="mr-1 h-3 w-3" /> {item.from.join(" + ")} → {item.perturbed}
            </Badge>
          ))}
        </div>

        {/* The mapping is what makes the two timelines comparable at all —
            pyannote's labels are arbitrary per run. */}
        <div className="mt-3 text-xs text-muted-foreground">
          Label alignment (Hungarian matching):{" "}
          {Object.keys(delta.speaker_mapping).length === 0 ? (
            <span className="text-foreground">no speaker matched between runs</span>
          ) : (
            <span className="font-mono text-foreground">
              {Object.entries(delta.speaker_mapping)
                .map(([perturbed, original]) => `${perturbed}→${original}`)
                .join(", ")}
            </span>
          )}
        </div>
      </div>

      {delta.boundary_shifts.shifts.length > 0 && (
        <div className="rounded-lg border p-3">
          <div className="mb-2 text-xs text-muted-foreground">
            Boundaries that moved more than {delta.boundary_shifts.threshold_seconds}s
          </div>
          <div className="space-y-1 font-mono text-xs">
            {delta.boundary_shifts.shifts.map((shift) => (
              <div key={`${shift.segment_id}-${shift.edge}`} className="flex items-center gap-2">
                <span className="text-muted-foreground">
                  {shift.segment_id} {shift.edge}
                </span>
                <span>{shift.original.toFixed(2)}s</span>
                <ArrowRight className="h-3 w-3 text-muted-foreground" />
                <span>{shift.perturbed.toFixed(2)}s</span>
                <span className={shift.delta > 0 ? "text-amber-600" : "text-blue-600"}>
                  ({shift.delta > 0 ? "+" : ""}
                  {shift.delta.toFixed(2)}s)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

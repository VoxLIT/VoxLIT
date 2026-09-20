import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { RangeSlider } from "@/components/ui/range-slider";
import { Slider } from "@/components/ui/slider";
import { Loader2, Play, Volume2, EyeOff } from "lucide-react";
import { PerturbationType } from "./types";

interface PerturbationControlsProps {
  activeType: PerturbationType;
  onActiveTypeChange: (type: PerturbationType) => void;
  /** Slider units: percent. Converted to the backend's 0-0.5 scale on send. */
  noisePercent: number;
  onNoisePercentChange: (value: number) => void;
  maskRange: [number, number];
  onMaskRangeChange: (value: [number, number]) => void;
  onRun: () => void;
  isRunning: boolean;
  disabled: boolean;
  disabledReason: string | null;
}

/** Slider percent -> the backend's `noise_level` (0 < x <= 0.5).
 *  5% on the slider is 0.025, which is around where the timeline starts to
 *  visibly move on the demo clip. */
export const noisePercentToLevel = (percent: number): number =>
  Math.round((percent / 100) * 0.5 * 10000) / 10000;

const TYPE_TABS: { id: PerturbationType; label: string; icon: typeof Volume2 }[] = [
  { id: "noise", label: "Gaussian noise", icon: Volume2 },
  { id: "time_masking", label: "Time masking", icon: EyeOff },
];

export const PerturbationControls = ({
  activeType,
  onActiveTypeChange,
  noisePercent,
  onNoisePercentChange,
  maskRange,
  onMaskRangeChange,
  onRun,
  isRunning,
  disabled,
  disabledReason,
}: PerturbationControlsProps) => {
  const noiseLevel = noisePercentToLevel(noisePercent);

  return (
    <div className="space-y-4">
      {/* One perturbation at a time: the diff is a two-way comparison, so
          stacking transforms would make it impossible to attribute a change. */}
      <div className="flex flex-wrap gap-2">
        {TYPE_TABS.map(({ id, label, icon: Icon }) => (
          <Button
            key={id}
            type="button"
            size="sm"
            variant={activeType === id ? "default" : "outline"}
            onClick={() => onActiveTypeChange(id)}
            disabled={isRunning}
          >
            <Icon className="mr-2 h-3.5 w-3.5" />
            {label}
          </Button>
        ))}
      </div>

      {activeType === "noise" ? (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Noise level</span>
            <Badge variant="outline" className="font-mono text-xs">
              {noisePercent}% — noise_level {noiseLevel}
            </Badge>
          </div>
          <Slider
            value={[noisePercent]}
            onValueChange={([value]) => onNoisePercentChange(value)}
            min={1}
            max={100}
            step={1}
            disabled={isRunning}
          />
          <p className="text-xs text-muted-foreground">
            Gaussian noise added to every sample. Low levels usually leave the
            timeline untouched — push it up until the pipeline starts splitting
            or losing turns.
          </p>
        </div>
      ) : (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs text-muted-foreground">Masked region</span>
            <Badge variant="outline" className="font-mono text-xs">
              {maskRange[0]}% – {maskRange[1]}%
            </Badge>
          </div>
          <RangeSlider
            value={maskRange}
            onValueChange={onMaskRangeChange}
            min={0}
            max={100}
            step={1}
            showLabels={false}
            disabled={isRunning}
          />
          <p className="text-xs text-muted-foreground">
            Silences that span of the recording. Mask a speaker's only turn and
            they should drop out of the timeline entirely.
          </p>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2">
        <Button onClick={onRun} disabled={disabled || isRunning}>
          {isRunning ? (
            <>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Re-diarizing
              perturbed audio… (minutes on a full meeting)
            </>
          ) : (
            <>
              <Play className="mr-2 h-4 w-4" /> Run perturbation
            </>
          )}
        </Button>
        {!isRunning && disabledReason && (
          <span className="text-xs text-muted-foreground">{disabledReason}</span>
        )}
      </div>
    </div>
  );
};

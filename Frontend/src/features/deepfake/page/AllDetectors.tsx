import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import { motion } from "motion/react";
import { Flame, Layers, Loader2, Play, Scissors, Sparkles } from "lucide-react";
import type { DeepfakeResult, DeepfakeSaliency, RecordingInfo, SilenceProbeResult } from "../types";
import { readSaliencyVerdict } from "../SaliencyPanel";
import { readProbeVerdict } from "../SilenceProbeCard";
import { errorMessage, isUserClip, postDeepfake } from "./api";
import { loadResult, markRevealed, readRevealed, resultKey } from "./session";
import { leanWords, formatScore } from "./palette";
import { usePalette } from "./theme";
import { ErrorNote, PrimaryButton, VerdictChip } from "./ui";

export type Feature = "run" | "silence-probe" | "saliency";
const FEATURES: Feature[] = ["run", "silence-probe", "saliency"];

interface ModelOption {
  id: string;
  label: string;
}

interface Cell {
  run?: DeepfakeResult;
  "silence-probe"?: SilenceProbeResult;
  saliency?: DeepfakeSaliency;
  pending: Feature[];
  errors: Partial<Record<Feature, string>>;
}

const emptyCell = (): Cell => ({ pending: [], errors: {} });
const shortLabel = (label: string) => label.replace(/\s*\(Model [A-Z]\)/, "");
const modelLetter = (label: string) => label.match(/\(Model ([A-Z])\)/)?.[1];

interface AllDetectorsProps {
  models: ModelOption[];
  recording: RecordingInfo | null;
}

/**
 * The same clip through every detector and every per-clip feature, side by
 * side: the verdict, the silence test and the listening heatmap. Detectors
 * disagree often — that disagreement is the most honest thing this page can
 * show about a clip nobody has labelled, like the visitor's own recording.
 *
 * Jobs run one at a time: each detector is a large model, and loading them
 * all at once would compete for the same memory.
 */
export const AllDetectors = ({ models, recording }: AllDetectorsProps) => {
  const { REAL, FAKE } = usePalette();
  const recordingId = recording?.recording_id ?? "";
  // Results revealed for this clip, here or in the single-detector panels
  // (they share storage keys), come back after a refresh. Stored results that
  // are not revealed wait for their button.
  const restore = (): Record<string, Cell> => {
    if (!recordingId) return {};
    const restored: Record<string, Cell> = {};
    for (const option of models) {
      const cell: Cell = emptyCell();
      let found = false;
      for (const feature of FEATURES) {
        const stored = readRevealed<unknown>(resultKey(feature, option.id, recordingId));
        if (stored) {
          (cell as unknown as Record<string, unknown>)[feature] = stored;
          found = true;
        }
      }
      if (found) restored[option.id] = cell;
    }
    return restored;
  };
  const [cells, setCells] = useState<Record<string, Cell>>(restore);
  const [runningAll, setRunningAll] = useState(false);
  const generation = useRef(0);

  useEffect(() => {
    generation.current += 1;
    setCells(restore());
    setRunningAll(false);
    // restore reads recordingId and the model list; the clip is what matters.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [recordingId]);

  const patch = (model: string, update: (cell: Cell) => Cell) =>
    setCells((current) => ({ ...current, [model]: update(current[model] ?? emptyCell()) }));

  const runOne = useCallback(
    async (model: string, feature: Feature, token = generation.current): Promise<boolean> => {
      if (!recordingId) return false;
      patch(model, (cell) => ({
        ...cell,
        pending: [...cell.pending.filter((item) => item !== feature), feature],
        errors: { ...cell.errors, [feature]: undefined },
      }));
      try {
        const key = resultKey(feature, model, recordingId);
        const { payload } = await loadResult(key, () =>
          postDeepfake<unknown>(feature, { model, recording_id: recordingId }),
        );
        if (token !== generation.current) return false;
        markRevealed(key);
        patch(model, (cell) => ({ ...cell, [feature]: payload, pending: cell.pending.filter((item) => item !== feature) }));
        return true;
      } catch (caught) {
        if (token !== generation.current) return false;
        patch(model, (cell) => ({
          ...cell,
          pending: cell.pending.filter((item) => item !== feature),
          errors: { ...cell.errors, [feature]: errorMessage(caught, "Failed.") },
        }));
        return false;
      }
    },
    [recordingId],
  );

  const runAll = async () => {
    const token = generation.current;
    setRunningAll(true);
    // Detector by detector, so each model is loaded once and then reused
    // for all three of its features before the next one is brought in.
    for (const model of models) {
      for (const feature of FEATURES) {
        if (token !== generation.current) return;
        if (cells[model.id]?.[feature]) continue;
        // A detector that cannot load (e.g. a gated checkpoint) would fail
        // every feature with the same message — say it once, move on.
        if (!(await runOne(model.id, feature, token))) break;
      }
    }
    if (token === generation.current) setRunningAll(false);
  };

  if (!recording) return null;

  const verdicts = models.map((model) => cells[model.id]?.run).filter((result): result is DeepfakeResult => !!result);
  const synthetic = verdicts.filter((result) => result.decision === "spoof").length;
  const complete = models.every((model) => FEATURES.every((feature) => cells[model.id]?.[feature]));

  return (
    <div className="space-y-5">
      <div className="df-glass flex flex-wrap items-center gap-4 p-3">
        <div className="min-w-0 flex-1">
          <div className="text-[11px] uppercase tracking-widest text-slate-400">
            {isUserClip(recording.recording_id) ? "Your clip" : "Dataset clip"}
          </div>
          <div className="truncate font-mono text-base font-semibold text-white">{recording.display_filename}</div>
          {verdicts.length > 0 ? (
            <p className="mt-1 text-sm text-slate-300">
              <span className="font-semibold" style={{ color: synthetic > verdicts.length / 2 ? FAKE : REAL }}>
                {synthetic} of {verdicts.length}
              </span>{" "}
              detector{verdicts.length === 1 ? "" : "s"} {synthetic === 1 ? "says" : "say"} synthetic
              {verdicts.length === models.length && (synthetic === 0 || synthetic === models.length)
                ? ". They all agree."
                : verdicts.length === models.length
                  ? ". They disagree, so no single score should be trusted on its own."
                  : "."}
            </p>
          ) : (
            <p className="mt-1 text-sm text-slate-400">
              Run all {models.length} detectors through every feature, or pick single cells below.
            </p>
          )}
        </div>
        {!complete && (
          <PrimaryButton onClick={runAll} busy={runningAll}>
            <Layers className="h-4 w-4" /> {runningAll ? "Running the detectors" : `Run everything on all ${models.length}`}
          </PrimaryButton>
        )}
      </div>
      {runningAll && (
        <p className="-mt-3 text-center text-xs text-slate-500">
          One job at a time. A detector&apos;s first run downloads it, which can take a few minutes.
        </p>
      )}

      <div className="grid gap-5 lg:grid-cols-3">
        {models.map((model, index) => (
          <motion.div
            key={model.id}
            initial={{ opacity: 0, y: 24 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: index * 0.08 }}
            className="df-glass flex min-w-0 flex-col gap-3 p-3"
            data-testid="detector-card"
          >
            <div className="flex items-center gap-3">
              {modelLetter(model.label) && (
                <span className="grid h-8 w-8 place-items-center rounded-xl bg-white/10 font-display text-sm font-bold text-white">
                  {modelLetter(model.label)}
                </span>
              )}
              <div className="font-display text-base font-semibold text-white">{shortLabel(model.label)}</div>
            </div>
            <ModelColumn cell={cells[model.id] ?? emptyCell()} onRun={(feature) => runOne(model.id, feature)} />
          </motion.div>
        ))}
      </div>
    </div>
  );
};

const ModelColumn = ({ cell, onRun }: { cell: Cell; onRun: (feature: Feature) => void }) => (
  <>
    <Block
      icon={<Sparkles className="h-4 w-4 text-violet-300" />}
      title="Verdict"
      action="Score it"
      pending={cell.pending.includes("run")}
      error={cell.errors.run}
      onRun={() => onRun("run")}
    >
      {cell.run && <VerdictBlock result={cell.run} />}
    </Block>
    <Block
      icon={<Scissors className="h-4 w-4 text-cyan-300" />}
      title="Voice or silence?"
      action="Run the silence test"
      pending={cell.pending.includes("silence-probe")}
      error={cell.errors["silence-probe"]}
      onRun={() => onRun("silence-probe")}
    >
      {cell["silence-probe"] && <SilenceBlock result={cell["silence-probe"]} />}
    </Block>
    <Block
      icon={<Flame className="h-4 w-4 text-rose-300" />}
      title="Where did it listen?"
      action="Show the heatmap"
      pending={cell.pending.includes("saliency")}
      error={cell.errors.saliency}
      onRun={() => onRun("saliency")}
    >
      {cell.saliency && <HeatBlock result={cell.saliency} />}
    </Block>
  </>
);

const Block = ({
  icon,
  title,
  action,
  pending,
  error,
  onRun,
  children,
}: {
  icon: ReactNode;
  title: string;
  action: string;
  pending: boolean;
  error?: string;
  onRun: () => void;
  children: ReactNode;
}) => (
  <section className="space-y-2 rounded-2xl bg-white/[0.03] p-3 ring-1 ring-white/10" aria-label={title}>
    <div className="flex items-center gap-2 text-sm font-semibold text-white">
      {icon} {title}
    </div>
    {children ||
      (pending ? (
        <div className="flex items-center gap-2 py-2 text-xs text-slate-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Working
        </div>
      ) : (
        <button
          type="button"
          onClick={onRun}
          className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold text-slate-200 ring-1 ring-white/15 transition hover:bg-white/10"
        >
          <Play className="h-3 w-3" /> {action}
        </button>
      ))}
    {error && (
      <div className="min-w-0 [overflow-wrap:anywhere]">
        <ErrorNote>{error}</ErrorNote>
      </div>
    )}
  </section>
);

const Bar = ({ score, threshold }: { score: number; threshold: number }) => {
  const { scoreColor } = usePalette();
  return (
    <div className="relative h-2 w-full overflow-hidden rounded-full bg-white/10">
      <motion.div
        className="h-full rounded-full"
        style={{ background: scoreColor(score) }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.max(2, score * 100)}%` }}
        transition={{ duration: 0.8, ease: [0.2, 0.7, 0.3, 1] }}
      />
      <span className="absolute inset-y-0 w-px bg-white/70" style={{ left: `${threshold * 100}%` }} aria-hidden />
    </div>
  );
};

const VerdictBlock = ({ result }: { result: DeepfakeResult }) => {
  const { scoreColor } = usePalette();
  const spoof = result.decision === "spoof";
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-2">
        <VerdictChip spoof={spoof} size="sm" />
        <span className="font-mono text-lg font-semibold" style={{ color: scoreColor(result.spoof_probability) }}>
          {formatScore(result.spoof_probability)}
        </span>
      </div>
      <Bar score={result.spoof_probability} threshold={result.threshold} />
      <p className="text-xs text-slate-400">
        {leanWords(result.spoof_probability, result.threshold)} {spoof ? "synthetic" : "real"}, threshold{" "}
        {result.threshold.toFixed(2)}
        {result.truncated ? `, first ${result.analysed_seconds}s scored` : ""}
      </p>
    </div>
  );
};

const LEGS: { key: keyof SilenceProbeResult["variants"]; label: string }[] = [
  { key: "original", label: "Whole clip" },
  { key: "trimmed", label: "Silence cut" },
  { key: "non_speech", label: "Only silence" },
];

const SilenceBlock = ({ result }: { result: SilenceProbeResult }) => {
  const verdict = readProbeVerdict(result);
  return (
    <div className="space-y-2">
      {LEGS.map((leg) => {
        const variant = result.variants[leg.key];
        const score = variant.applicable ? variant.spoof_probability : null;
        return (
          <div key={leg.key} className="grid grid-cols-[84px_1fr_72px] items-center gap-2 text-xs" title={variant.reason}>
            <span className="text-slate-400">{leg.label}</span>
            {score !== null ? (
              <Bar score={score} threshold={result.threshold} />
            ) : (
              <span className="font-mono text-slate-500">{variant.seconds.toFixed(3)} s, no score</span>
            )}
            <span className="text-right font-mono text-white">
              {score !== null ? `${formatScore(score)}${variant.reliable === false ? " †" : ""}` : "none"}
            </span>
          </div>
        );
      })}
      {LEGS.some((leg) => result.variants[leg.key].applicable && result.variants[leg.key].reliable === false) && (
        <p className="text-[10px] text-slate-400">† scored on under {result.min_non_speech_seconds.toFixed(2)} s of audio; indicative only.</p>
      )}
      {verdict && (
        <p className={`text-xs font-semibold ${verdict.alarming ? "text-amber-300" : "text-emerald-300"}`}>{verdict.title}</p>
      )}
    </div>
  );
};

const HeatBlock = ({ result }: { result: DeepfakeSaliency }) => {
  const { REAL, FAKE } = usePalette();
  const verdict = readSaliencyVerdict(result);
  const duration = result.total_duration || 1;
  return (
    <div className="space-y-2">
      <svg viewBox="0 0 300 36" className="h-9 w-full" role="img" aria-label="Attribution over time">
        {result.speech_intervals.map(([start, end], index) => (
          <rect key={`s${index}`} x={(start / duration) * 300} width={((end - start) / duration) * 300} y={30} height={6} fill={REAL} opacity={0.5} />
        ))}
        {result.segments.map((segment, index) => (
          <rect
            key={index}
            x={(segment.start_time / duration) * 300}
            width={Math.max(0.5, ((segment.end_time - segment.start_time) / duration) * 300)}
            y={0}
            height={28}
            fill={FAKE}
            opacity={0.08 + 0.92 * segment.intensity}
          />
        ))}
      </svg>
      <p className="text-[11px] text-slate-500">Red: moments that moved the verdict most. Blue underline: speech.</p>
      {verdict && (
        <p className={`text-xs font-semibold ${verdict.alarming ? "text-amber-300" : "text-emerald-300"}`}>{verdict.title}</p>
      )}
    </div>
  );
};

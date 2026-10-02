import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import { BarChart3, FlaskConical, SlidersHorizontal } from "lucide-react";
import type { DeepfakeEvaluation, DetPoint, EvaluationCondition } from "../types";
import { errorMessage, postDeepfake } from "./api";
import { readSession, resultKey, writeSession } from "./session";
import { describeAttack } from "./attacks";
import { AttackBars, CountUp, DetChart, Histogram } from "./charts";
import { usePalette } from "./theme";
import { Disclosure, ErrorNote, FeatureImage, PrimaryButton } from "./ui";

const CONDITION_OPTIONS: { id: EvaluationCondition; label: string; hint: string }[] = [
  {
    id: "silence_trimmed",
    label: "Silence trimmed",
    hint: "Leading and trailing non-speech removed before scoring: the detector must judge the voice itself.",
  },
  {
    id: "as_distributed",
    label: "As distributed",
    hint: "Clips exactly as ASVspoof ships them. Genuine clips carry far more silence, which detectors can exploit.",
  },
];

const pct = (fraction: number, digits = 2) => `${(fraction * 100).toFixed(digits)}%`;

/** Wilson score interval: the binomial interval used for rates at a cut the
 *  visitor drags (the server reports exact Clopper-Pearson ones at τ_op). */
const wilson = (errors: number, n: number, z = 1.959964): [number, number] => {
  if (n === 0) return [0, 1];
  const p = errors / n;
  const denominator = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denominator;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denominator;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
};

/**
 * Feature 1 for everyone: "how good is this detector overall?" The whole
 * labelled subset is scored; the visitor drags the cut and watches the two
 * kinds of mistake trade off. Aggregates only — no clip's label is shown.
 */
export const DetectorReport = ({
  model,
  modelLabel,
  datasetSize,
  customDataset = null,
  labelledCount = null,
}: {
  model: string;
  modelLabel: string;
  datasetSize: number;
  /** A researcher's own dataset name, or null for the built-in subset. */
  customDataset?: string | null;
  /** For a custom dataset: how many of its files a label file matched. */
  labelledCount?: number | null;
}) => {
  const { REAL, MID, FAKE } = usePalette();
  const datasetAvailable = datasetSize > 0 && (customDataset === null || (labelledCount ?? 0) > 0);
  const [condition, setConditionState] = useState<EvaluationCondition>(
    () => readSession<EvaluationCondition>("report.condition") ?? "silence_trimmed",
  );
  const setCondition = (value: EvaluationCondition) => {
    setConditionState(value);
    writeSession("report.condition", value);
  };
  const datasetPart = customDataset ? `custom-${customDataset}` : "builtin";
  const keyFor = (which: EvaluationCondition) => resultKey("scores.v2", model, datasetPart, which);
  const key = keyFor(condition);
  const otherCondition: EvaluationCondition = condition === "silence_trimmed" ? "as_distributed" : "silence_trimmed";
  const [other, setOther] = useState<DeepfakeEvaluation | null>(() => readSession<DeepfakeEvaluation>(keyFor(otherCondition)));
  const [result, setResult] = useState<DeepfakeEvaluation | null>(() => readSession<DeepfakeEvaluation>(key));
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cut, setCutState] = useState(() => readSession<number>(`${key}.cut`) ?? 0.5);
  const setCut = (value: number) => {
    setCutState(value);
    writeSession(`${key}.cut`, value);
  };

  // Scoring the whole subset takes minutes; a reply that arrives after the
  // model changed belongs to the old model and is dropped.
  const latestRequest = useRef(0);

  const body = (which: EvaluationCondition) => ({
    model,
    condition: which,
    ...(customDataset ? { dataset: customDataset } : {}),
  });

  const run = async () => {
    const request = ++latestRequest.current;
    setRunning(true);
    setError(null);
    try {
      const payload = await postDeepfake<DeepfakeEvaluation>("scores", body(condition));
      if (request !== latestRequest.current) return;
      setResult(payload);
      writeSession(key, payload);
      setCut(payload.operating_point.threshold);
    } catch (caught) {
      if (request === latestRequest.current) setError(errorMessage(caught, "Evaluation failed."));
    } finally {
      if (request === latestRequest.current) setRunning(false);
    }
    // The other condition, for the side-by-side comparison. Best effort.
    if (request !== latestRequest.current || readSession(keyFor(otherCondition))) return;
    postDeepfake<DeepfakeEvaluation>("scores", body(otherCondition))
      .then((payload) => {
        writeSession(keyFor(otherCondition), payload);
        if (request === latestRequest.current) setOther(payload);
      })
      .catch(() => undefined);
  };

  // A report belongs to one model, dataset and condition.
  useEffect(() => {
    latestRequest.current += 1;
    setResult(readSession<DeepfakeEvaluation>(key));
    setOther(readSession<DeepfakeEvaluation>(keyFor(otherCondition)));
    setCutState(readSession<number>(`${key}.cut`) ?? 0.5);
    setError(null);
    setRunning(false);
    // Switching condition after a report exists fetches the new one directly
    // instead of falling back to the start screen.
    if (!readSession(key) && readSession(keyFor(otherCondition))) void run();
    // run/keyFor read only values that `key` already encodes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const conditionSwitch = (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-xs font-medium text-slate-400">Evaluation condition</span>
      <div className="inline-flex rounded-sm p-0.5 ring-1 ring-white/15" role="radiogroup" aria-label="Evaluation condition">
        {CONDITION_OPTIONS.map((option) => (
          <button
            key={option.id}
            type="button"
            role="radio"
            aria-checked={condition === option.id}
            title={option.hint}
            onClick={() => setCondition(option.id)}
            disabled={running}
            className="rounded-sm px-2.5 py-1 text-xs font-medium text-slate-400 transition hover:text-white aria-checked:bg-white/10 aria-checked:text-white disabled:opacity-60"
          >
            {option.label}
          </button>
        ))}
      </div>
      <span className="text-xs text-slate-500">{CONDITION_OPTIONS.find((option) => option.id === condition)?.hint}</span>
    </div>
  );

  const live = useMemo<DetPoint | null>(() => {
    if (!result || result.det_curve.length === 0) return null;
    // Decisions only change at a clip's score, so every cut behaves exactly
    // like the first evaluated threshold at or above it (s ≥ τ is spoof).
    const ordered = [...result.det_curve].sort((a, b) => a.threshold - b.threshold);
    return ordered.find((point) => point.threshold >= cut) ?? ordered[ordered.length - 1];
  }, [result, cut]);

  if (!result) {
    return (
      <div className="grid items-center gap-6 lg:grid-cols-[1.1fr_1fr]">
        <FeatureImage
          src="/deepfake/evaluation.jpg"
          alt="Visitors lining up for hearing tests at the 1939 New York World's Fair"
          caption="1939: visitors queue to test their hearing against the VODER speech machine. Here, the detector takes the test."
          className="h-64 lg:h-80"
        />
        <div className="space-y-4">
          <p className="text-lg text-slate-200">
            One clip can&apos;t tell you if a detector is any good. Run it on all{" "}
            <span className="font-semibold text-white">
              {(customDataset ? labelledCount : datasetSize) || "the"} labelled clips
            </span>{" "}
            and see how cleanly it separates real voices from synthetic ones, and what every threshold would cost.
          </p>
          {conditionSwitch}
          <ul className="space-y-2 text-sm text-slate-300">
            <li className="flex gap-2">
              <span className="mt-1.5 h-2 w-2 rounded-full" style={{ background: REAL }} /> Drag the cut and watch
              mistakes appear
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-2 w-2 rounded-full" style={{ background: MID }} /> Find the balance point
              where both errors are equal
            </li>
            <li className="flex gap-2">
              <span className="mt-1.5 h-2 w-2 rounded-full" style={{ background: FAKE }} /> See which voice
              generators fool it most
            </li>
          </ul>
          {!datasetAvailable && (
            <ErrorNote>
              {customDataset
                ? `"${customDataset}" has no label file yet. Add one in Manage Datasets (CSV "filename,label" or ASVspoof protocol lines) to measure EER on it.`
                : "The labelled ASVspoof subset isn't on this server. Build it with scripts/prepare_asvspoof_la_subset.py."}
            </ErrorNote>
          )}
          <PrimaryButton onClick={run} busy={running} disabled={!datasetAvailable}>
            <BarChart3 className="h-4 w-4" />
            {running ? "Testing on every clip" : `Test ${modelLabel}`}
          </PrimaryButton>
          {running && (
            <p className="text-xs text-slate-400">
              One pass per clip{condition === "silence_trimmed" ? " on the trimmed audio" : ""}. This takes about a minute
              the first time and is cached after.
            </p>
          )}
          {error && <ErrorNote>{error}</ErrorNote>}
        </div>
      </div>
    );
  }

  const fakesThrough = live ? Math.round(live.false_acceptance_rate * result.spoof_count) : 0;
  const realsFlagged = live ? Math.round(live.false_rejection_rate * result.bonafide_count) : 0;
  const hardest = result.per_attack
    .filter((row) => row.is_spoof)
    .reduce<typeof result.per_attack[number] | null>((low, row) => (!low || row.mean_score < low.mean_score ? row : low), null);
  const hardestInfo = hardest ? describeAttack(hardest.attack) : null;

  const zeroEer = result.eer_percent === 0;
  const eerHint = zeroEer
    ? `no errors; 95% upper bound ${result.eer_zero_upper_bound_percent?.toFixed(2) ?? "?"}% (exact binomial)`
    : result.eer_ci_percent
      ? `95% CI ${result.eer_ci_percent[0].toFixed(2)}–${result.eer_ci_percent[1].toFixed(2)}% (bootstrap, ${result.bootstrap_resamples ?? 1000}×)`
      : "equal error rate, lower is better";

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      {conditionSwitch}
      {/* headline */}
      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <BigStat
          label={`Equal error rate (EER), ${condition === "silence_trimmed" ? "silence trimmed" : "as distributed"}`}
          value={<CountUp value={result.eer_percent} format={(v) => `${v.toFixed(2)}%`} />}
          hint={eerHint}
          colour={MID}
        />
        <BigStat
          label="ROC-AUC (threshold-free)"
          value={<CountUp value={result.roc_auc ?? 0} format={(v) => v.toFixed(3)} />}
          hint="P(spoof clip outscores genuine clip)"
          colour={MID}
        />
        <BigStat
          label="Genuine clips tested"
          value={<CountUp value={result.bonafide_count} format={(v) => Math.round(v).toString()} />}
          hint="real human recordings"
          colour={REAL}
        />
        <BigStat
          label="Synthetic clips tested"
          value={<CountUp value={result.spoof_count} format={(v) => Math.round(v).toString()} />}
          hint="made by voice generators"
          colour={FAKE}
        />
      </div>

      {other && other.condition && <ConditionComparison current={result} other={other} />}

      {/* play with the cut */}
      <div className="df-glass p-3">
        <div className="mb-3 flex flex-wrap items-baseline gap-x-2 gap-y-1">
          <SlidersHorizontal className="h-4 w-4 self-center text-violet-300" />
          <h4 className="font-display text-lg font-semibold text-white">Score distributions by class</h4>
          <span className="text-sm text-slate-400">
            n = {result.bonafide_count} bona fide, {result.spoof_count} spoof. Clips with s ≥ τ are classified as spoof; hatched bins are misclassified.
          </span>
        </div>
        <Histogram
          bonafide={result.distributions.bonafide}
          spoof={result.distributions.spoof}
          threshold={cut}
          eerThreshold={result.eer_threshold}
          shippedThreshold={result.operating_point.threshold}
        />
        <div className="mt-2 flex items-center gap-3 px-1">
          <span className="shrink-0 text-xs font-medium text-slate-400">Threshold τ</span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.005}
            value={cut}
            onChange={(event) => setCut(Number(event.target.value))}
            aria-label="Decision threshold"
            aria-valuetext={`threshold ${cut.toFixed(3)}`}
            className="df-range"
          />
          <span className="w-12 shrink-0 text-right font-mono text-xs text-slate-300">{cut.toFixed(3)}</span>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          <button type="button" onClick={() => setCut(result.eer_threshold)} className="rounded-md px-2.5 py-1 font-medium ring-1 ring-white/20 transition hover:bg-white/5" style={{ color: MID }}>
            Set τ = τ<sub>EER</sub> ({result.eer_threshold.toFixed(3)})
          </button>
          <button type="button" onClick={() => setCut(result.operating_point.threshold)} className="rounded-md px-2.5 py-1 font-medium text-slate-300 ring-1 ring-white/20 transition hover:bg-white/5">
            Set τ = τ<sub>op</sub> ({result.operating_point.threshold.toFixed(3)})
          </button>
          <span className="ml-auto flex flex-wrap items-center gap-3 text-slate-400">
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-3.5 border" style={{ background: `${REAL}73`, borderColor: REAL }} /> Bona fide</span>
            <span className="flex items-center gap-1.5"><span className="h-2.5 w-3.5 border" style={{ background: `${FAKE}73`, borderColor: FAKE }} /> Spoof</span>
            <span className="flex items-center gap-1.5">
              <span className="h-2.5 w-3.5 border border-slate-800" style={{ background: "repeating-linear-gradient(45deg, rgba(15,23,42,.7) 0 1px, transparent 1px 4px)" }} /> Misclassified
            </span>
            <span className="flex items-center gap-1.5"><span className="w-4 border-t border-dashed" style={{ borderColor: MID }} /> <span>τ<sub>EER</sub></span></span>
            <span className="flex items-center gap-1.5"><span className="w-4 border-t border-dotted border-slate-500" /> <span>τ<sub>op</sub></span></span>
          </span>
        </div>

        {live && (
          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            <Mistake
              count={fakesThrough}
              total={result.spoof_count}
              text="fakes slip through as real"
              colour={FAKE}
            />
            <Mistake count={realsFlagged} total={result.bonafide_count} text="real voices wrongly flagged" colour={REAL} />
          </div>
        )}
        {live && (
          <MetricsTable
            cut={cut}
            fn={fakesThrough}
            fp={realsFlagged}
            spoofCount={result.spoof_count}
            bonafideCount={result.bonafide_count}
          />
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="df-glass p-3">
          <h4 className="font-display text-lg font-semibold text-white">Detection error trade-off (DET)</h4>
          <p className="mb-3 text-sm text-slate-400">
            FAR against FRR across every threshold. The circle marks the equal error rate, the square the current τ, and the dashed line FAR = FRR.
          </p>
          {live && <DetChart points={result.det_curve} live={live} threshold={cut} eerPercent={result.eer_percent} />}
        </div>
        <div className="df-glass p-3">
          <h4 className="font-display text-lg font-semibold text-white">Which voice generators fool it?</h4>
          <p className="mb-4 text-sm text-slate-400">
            Average score per generator. Shorter bars look more real to the detector.
            {hardest && (
              <>
                {" "}
                Hardest to catch here: <span className="font-mono text-white">{hardest.attack}</span>
                {hardestInfo && ` (${hardestInfo.model}, ${hardestInfo.vocoder} vocoder)`}
                .
              </>
            )}
          </p>
          <AttackBars rows={result.per_attack} />
        </div>
      </div>

      <Disclosure>
        <p>{result.threshold_provenance}</p>
        <dl className="grid grid-cols-2 gap-3 text-xs sm:grid-cols-4">
          <div><dt className="text-slate-400">EER</dt><dd className="font-mono text-white">{result.eer_percent.toFixed(2)}%</dd></div>
          {result.eer_ci_percent && (
            <div><dt className="text-slate-400">EER 95% CI (bootstrap)</dt><dd className="font-mono text-white">{result.eer_ci_percent[0].toFixed(2)}–{result.eer_ci_percent[1].toFixed(2)}%</dd></div>
          )}
          {result.eer_resolution_percent !== undefined && (
            <div><dt className="text-slate-400">EER resolution</dt><dd className="font-mono text-white">{result.eer_resolution_percent.toFixed(2)}% per error</dd></div>
          )}
          {result.roc_auc !== undefined && (
            <div><dt className="text-slate-400">ROC-AUC</dt><dd className="font-mono text-white">{result.roc_auc.toFixed(4)}</dd></div>
          )}
          <div><dt className="text-slate-400">EER threshold</dt><dd className="font-mono text-white">{result.eer_threshold.toFixed(4)}</dd></div>
          <div><dt className="text-slate-400">cut in use</dt><dd className="font-mono text-white">{result.operating_point.threshold.toFixed(2)}{result.operating_point.calibrated ? "" : " (uncalibrated)"}</dd></div>
          <div><dt className="text-slate-400">FAR / FRR at it</dt><dd className="font-mono text-white">{(result.operating_point.false_acceptance_rate * 100).toFixed(1)}% / {(result.operating_point.false_rejection_rate * 100).toFixed(1)}%</dd></div>
          {live && (
            <div className="col-span-2"><dt className="text-slate-400">at your cut ({cut.toFixed(3)})</dt><dd className="font-mono text-white">FAR {(live.false_acceptance_rate * 100).toFixed(1)}%, FRR {(live.false_rejection_rate * 100).toFixed(1)}%</dd></div>
          )}
          <div><dt className="text-slate-400">scored</dt><dd className="font-mono text-white">{result.scored} clips</dd></div>
          <div><dt className="text-slate-400">dataset</dt><dd className="font-mono text-white">{result.dataset_id}</dd></div>
          {result.condition_label && (
            <div className="col-span-2"><dt className="text-slate-400">condition</dt><dd className="font-mono text-white">{result.condition_label}</dd></div>
          )}
        </dl>
        {result.score_statistics && (
          <table className="w-full text-left text-xs">
            <caption className="mb-1 text-left text-slate-400">Score s by class (spoof probability)</caption>
            <thead className="text-slate-400">
              <tr>
                {["class", "n", "mean", "median", "SD", "min", "max"].map((head) => (
                  <th key={head} className="py-1 pr-3 font-medium">{head}</th>
                ))}
              </tr>
            </thead>
            <tbody className="font-mono text-white">
              {(["bonafide", "spoof"] as const).map((name) => {
                const stats = result.score_statistics![name];
                return (
                  <tr key={name} className="border-t border-white/10">
                    <td className="py-1 pr-3 font-sans">{name === "bonafide" ? "Bona fide" : "Spoof"}</td>
                    <td className="pr-3">{stats.count}</td>
                    {[stats.mean, stats.median, stats.standard_deviation, stats.minimum, stats.maximum].map((value, index) => (
                      <td key={index} className="pr-3">{value.toFixed(4)}</td>
                    ))}
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
        <p className="text-xs text-slate-400">
          Histogram bar heights use a square-root scale so small bins stay visible next to large ones. The DET curve
          defaults to normal-deviate (probit) axes, as in ASVspoof and NIST evaluations; error rates of zero cannot be
          placed on that scale and are drawn on a separate floor labelled 0, set off by an axis break. The EER CI is a
          stratified percentile bootstrap ({result.bootstrap_resamples ?? 1000} resamples, fixed seed); when the EER is 0
          the bootstrap degenerates, so the exact one-sided binomial bound (1 − 0.05<sup>1/n</sup>, ≈ 3/n) is reported
          instead. Rates at your cut use Wilson 95% intervals. &ldquo;Acceptance&rdquo;
          means accepted as genuine (the ASVspoof convention). Rates at your
          cut are exact: decisions only change at a clip&rsquo;s score, so they equal those at the next evaluated threshold. Generator ids are the ASVspoof 2019 LA attack systems (Wang et al., 2020): TTS is text-to-speech, VC is voice conversion, and TTS + VC converts TTS output; the label under each id names its waveform generator. A16 and A19 reuse training-set systems, the other eleven are unseen in training.
        </p>
      </Disclosure>
    </motion.div>
  );
};

const BigStat = ({ label, value, hint, colour }: { label: string; value: ReactNode; hint: string; colour: string }) => (
  <motion.div
    initial={{ opacity: 0, y: 12 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    className="df-glass p-3"
  >
    <div className="text-xs text-slate-400">{label}</div>
    <div className="font-mono text-2xl font-semibold tabular-nums" style={{ color: colour }}>
      {value}
    </div>
    <div className="text-xs text-slate-500">{hint}</div>
  </motion.div>
);

const Mistake = ({ count, total, text, colour }: { count: number; total: number; text: string; colour: string }) => {
  const { WARN, ink } = usePalette();
  return (
    <div className="flex items-center gap-4 rounded-2xl bg-white/[0.03] p-4 ring-1 ring-white/10">
      <div className="font-display text-4xl font-semibold tabular-nums" style={{ color: count > 0 ? WARN : colour }}>
        <CountUp value={count} format={(v) => Math.round(v).toString()} />
      </div>
      <div className="text-sm text-slate-300">
        of {total} {text}
        <div className="mt-1.5 flex flex-wrap gap-0.5" aria-hidden>
          {Array.from({ length: total }, (_, index) => (
            <motion.span
              key={index}
              className="h-1.5 w-1.5 rounded-full"
              animate={{ backgroundColor: index < count ? WARN : ink(0.12), scale: index < count ? 1.25 : 1 }}
              transition={{ duration: 0.25 }}
            />
          ))}
        </div>
      </div>
    </div>
  );
};

/** Both evaluation conditions side by side: the gap is the silence shortcut. */
const ConditionComparison = ({ current, other }: { current: DeepfakeEvaluation; other: DeepfakeEvaluation }) => {
  const trimmed = current.condition === "silence_trimmed" ? current : other;
  const original = current.condition === "silence_trimmed" ? other : current;
  const gap = trimmed.eer_percent - original.eer_percent;
  const smallestClass = Math.min(trimmed.bonafide_count, trimmed.spoof_count);
  // Below ~30 clips a class, the two EERs' intervals are too wide to compare.
  const tooFew = smallestClass < 30;
  const describe = (report: DeepfakeEvaluation) =>
    report.eer_percent === 0
      ? `0.00% (≤ ${report.eer_zero_upper_bound_percent?.toFixed(2)}% at 95%)`
      : `${report.eer_percent.toFixed(2)}% [${report.eer_ci_percent?.[0].toFixed(2)}, ${report.eer_ci_percent?.[1].toFixed(2)}]`;
  return (
    <div className="df-glass grid gap-3 p-3 md:grid-cols-[auto_1fr] md:items-center">
      <div className="flex items-center gap-2">
        <FlaskConical className="h-4 w-4 text-cyan-300" />
        <span className="text-sm font-semibold text-white">Silence ablation</span>
      </div>
      <div className="grid gap-2 text-xs sm:grid-cols-3">
        <div>
          <div className="text-slate-400">EER as distributed</div>
          <div className="font-mono text-white">{describe(original)}</div>
          <div className="text-slate-500">AUC {original.roc_auc?.toFixed(3)}</div>
        </div>
        <div>
          <div className="text-slate-400">EER silence trimmed</div>
          <div className="font-mono text-white">{describe(trimmed)}</div>
          <div className="text-slate-500">AUC {trimmed.roc_auc?.toFixed(3)}</div>
        </div>
        <p className="text-xs leading-relaxed text-slate-300">
          {tooFew
            ? `With only ${smallestClass} labelled clips in the smaller class, the two conditions cannot be told apart reliably: compare the intervals, not the point values. Add more labelled clips for a conclusive ablation.`
            : gap > 1
            ? `Removing only the outer silence raises the EER by ${gap.toFixed(2)} points. Part of the as-distributed score comes from silence length, a known ASVspoof 2019 LA artefact (Müller et al., 2021), so the trimmed figure is the one to quote.`
            : "Trimming the silence barely moves the EER: this detector is not leaning on silence length here."}
        </p>
      </div>
    </div>
  );
};

/** The confusion matrix and the usual rates at the visitor's cut (spoof = positive). */
const MetricsTable = ({
  cut,
  fn,
  fp,
  spoofCount,
  bonafideCount,
}: {
  cut: number;
  fn: number;
  fp: number;
  spoofCount: number;
  bonafideCount: number;
}) => {
  const tp = spoofCount - fn;
  const tn = bonafideCount - fp;
  const precision = tp + fp ? tp / (tp + fp) : 0;
  const recall = spoofCount ? tp / spoofCount : 0;
  const specificity = bonafideCount ? tn / bonafideCount : 0;
  const f1 = precision + recall ? (2 * precision * recall) / (precision + recall) : 0;
  const far = wilson(fn, spoofCount);
  const frr = wilson(fp, bonafideCount);
  const rows: [string, string][] = [
    ["Accuracy", pct((tp + tn) / Math.max(1, spoofCount + bonafideCount))],
    ["Balanced accuracy", pct((recall + specificity) / 2)],
    ["Precision (spoof)", pct(precision)],
    ["Recall / TPR (spoof)", pct(recall)],
    ["Specificity / TNR", pct(specificity)],
    ["F1 (spoof)", f1.toFixed(4)],
    ["FAR, 95% Wilson", `${pct(fn / Math.max(1, spoofCount))} [${pct(far[0])}, ${pct(far[1])}]`],
    ["FRR, 95% Wilson", `${pct(fp / Math.max(1, bonafideCount))} [${pct(frr[0])}, ${pct(frr[1])}]`],
  ];
  return (
    <div className="mt-4 grid gap-4 border-t border-white/10 pt-3 lg:grid-cols-[auto_1fr]">
      <table className="text-center text-xs" aria-label={`Confusion matrix at threshold ${cut.toFixed(3)}`}>
        <caption className="mb-1 text-left text-slate-400">Confusion matrix at τ = {cut.toFixed(3)}</caption>
        <thead>
          <tr className="text-slate-400">
            <th />
            <th className="px-3 py-1 font-medium">pred. spoof</th>
            <th className="px-3 py-1 font-medium">pred. bona fide</th>
          </tr>
        </thead>
        <tbody className="font-mono text-white">
          <tr>
            <th className="pr-3 text-left font-sans font-medium text-slate-400">spoof</th>
            <td className="border border-white/10 px-3 py-1.5">TP {tp}</td>
            <td className="border border-white/10 px-3 py-1.5">FN {fn}</td>
          </tr>
          <tr>
            <th className="pr-3 text-left font-sans font-medium text-slate-400">bona fide</th>
            <td className="border border-white/10 px-3 py-1.5">FP {fp}</td>
            <td className="border border-white/10 px-3 py-1.5">TN {tn}</td>
          </tr>
        </tbody>
      </table>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs sm:grid-cols-4">
        {rows.map(([label, value]) => (
          <div key={label}>
            <dt className="text-slate-400">{label}</dt>
            <dd className="font-mono text-white">{value}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
};

import { type ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { motion } from "motion/react";
import { BarChart3, SlidersHorizontal } from "lucide-react";
import type { DeepfakeEvaluation, DetPoint } from "../types";
import { errorMessage, postDeepfake } from "./api";
import { AttackBars, CountUp, DetChart, Histogram } from "./charts";
import { usePalette } from "./theme";
import { Disclosure, ErrorNote, FeatureImage, PrimaryButton } from "./ui";

/**
 * Feature 1 for everyone: "how good is this detector overall?" The whole
 * labelled subset is scored; the visitor drags the cut and watches the two
 * kinds of mistake trade off. Aggregates only — no clip's label is shown.
 */
export const DetectorReport = ({
  model,
  modelLabel,
  datasetSize,
}: {
  model: string;
  modelLabel: string;
  datasetSize: number;
}) => {
  const { REAL, MID, FAKE } = usePalette();
  const datasetAvailable = datasetSize > 0;
  const [result, setResult] = useState<DeepfakeEvaluation | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [cut, setCut] = useState(0.5);

  // Scoring the whole subset takes minutes; a reply that arrives after the
  // model changed belongs to the old model and is dropped.
  const latestRequest = useRef(0);

  const run = async () => {
    const request = ++latestRequest.current;
    setRunning(true);
    setError(null);
    try {
      const payload = await postDeepfake<DeepfakeEvaluation>("scores", { model });
      if (request !== latestRequest.current) return;
      setResult(payload);
      setCut(payload.operating_point.threshold);
    } catch (caught) {
      if (request === latestRequest.current) setError(errorMessage(caught, "Evaluation failed."));
    } finally {
      if (request === latestRequest.current) setRunning(false);
    }
  };

  // A report belongs to one model.
  useEffect(() => {
    latestRequest.current += 1;
    setResult(null);
    setError(null);
    setRunning(false);
  }, [model]);

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
            <span className="font-semibold text-white">{datasetSize || "the"} labelled clips</span> and see how cleanly it separates
            real voices from synthetic ones, and what every threshold would cost.
          </p>
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
              The labelled ASVspoof subset isn&apos;t on this server. Build it with
              scripts/prepare_asvspoof_la_subset.py.
            </ErrorNote>
          )}
          <PrimaryButton onClick={run} busy={running} disabled={!datasetAvailable}>
            <BarChart3 className="h-4 w-4" />
            {running ? "Testing on every clip" : `Test ${modelLabel}`}
          </PrimaryButton>
          {running && <p className="text-xs text-slate-400">One pass per clip. This takes a few minutes the first time and is cached after.</p>}
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

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      {/* headline */}
      <div className="grid gap-3 sm:grid-cols-3">
        <BigStat
          label="Mistakes at its best balance point"
          value={<CountUp value={result.eer_percent} format={(v) => `${v.toFixed(1)}%`} />}
          hint="equal error rate, lower is better"
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
                Hardest to catch here: <span className="font-mono text-white">{hardest.attack}</span>.
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
          <div><dt className="text-slate-400">EER threshold</dt><dd className="font-mono text-white">{result.eer_threshold.toFixed(4)}</dd></div>
          <div><dt className="text-slate-400">cut in use</dt><dd className="font-mono text-white">{result.operating_point.threshold.toFixed(2)}{result.operating_point.calibrated ? "" : " (uncalibrated)"}</dd></div>
          <div><dt className="text-slate-400">FAR / FRR at it</dt><dd className="font-mono text-white">{(result.operating_point.false_acceptance_rate * 100).toFixed(1)}% / {(result.operating_point.false_rejection_rate * 100).toFixed(1)}%</dd></div>
          {live && (
            <div className="col-span-2"><dt className="text-slate-400">at your cut ({cut.toFixed(3)})</dt><dd className="font-mono text-white">FAR {(live.false_acceptance_rate * 100).toFixed(1)}%, FRR {(live.false_rejection_rate * 100).toFixed(1)}%</dd></div>
          )}
          <div><dt className="text-slate-400">scored</dt><dd className="font-mono text-white">{result.scored} clips</dd></div>
          <div><dt className="text-slate-400">dataset</dt><dd className="font-mono text-white">{result.dataset_id}</dd></div>
        </dl>
        <p className="text-xs text-slate-400">
          Histogram bar heights use a square-root scale so small bins stay visible next to large ones. The DET curve
          defaults to normal-deviate (probit) axes, as in ASVspoof and NIST evaluations; error rates of zero cannot be
          placed on that scale and are drawn on the axis floor, half the smallest non-zero rate. &ldquo;Acceptance&rdquo;
          means accepted as genuine (the ASVspoof convention). Rates at your
          cut are exact: decisions only change at a clip&rsquo;s score, so they equal those at the next evaluated threshold. Generator ids are ASVspoof 2019 LA attack systems.
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

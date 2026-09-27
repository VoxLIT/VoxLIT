import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { Scissors } from "lucide-react";
import type { ProbeVariant, SilenceProbeResult } from "../types";
import { readProbeVerdict } from "../SilenceProbeCard";
import { errorMessage, postDeepfake } from "./api";
import { usePalette } from "./theme";
import { Disclosure, ErrorNote, FeatureImage, Finding, PrimaryButton } from "./ui";
import { formatScore } from "./palette";

const LEGS: { key: keyof SilenceProbeResult["variants"]; label: string; hint: string }[] = [
  { key: "original", label: "Whole clip", hint: "as submitted" },
  { key: "trimmed", label: "Silence cut", hint: "outer silence removed" },
  { key: "non_speech", label: "Only silence", hint: "the voice removed" },
];

/**
 * Feature 2 for everyone: "is it listening to the voice, or to the silence?"
 * Three animated columns carry the answer; the energy threshold and seconds
 * are one click deeper.
 */
export const SilenceTest = ({ model, recordingId }: { model: string; recordingId: string }) => {
  const { REAL } = usePalette();
  const [result, setResult] = useState<SilenceProbeResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setResult(null);
    setError(null);
  }, [recordingId, model]);

  const run = async () => {
    setRunning(true);
    setError(null);
    try {
      setResult(await postDeepfake<SilenceProbeResult>("silence-probe", { model, recording_id: recordingId }));
    } catch (caught) {
      setError(errorMessage(caught, "The silence test failed."));
    } finally {
      setRunning(false);
    }
  };

  const verdict = result ? readProbeVerdict(result) : null;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Scissors className="h-4 w-4 text-cyan-300" />
        <h4 className="font-display text-base font-semibold text-white">Voice or silence?</h4>
      </div>

      {!result && (
        <>
          <FeatureImage
            src="/deepfake/silence-probe.png"
            alt="A speech waveform with its spectrogram and the pauses between words"
            caption="Real recordings carry more silence than most synthesizers. Does the detector lean on that?"
            className="h-32"
          />
          <p className="text-sm text-slate-300">
            We score the clip three times: whole, with the silence cut away, and with only the silence left.
          </p>
          <PrimaryButton onClick={run} busy={running} className="w-full">
            <Scissors className="h-4 w-4" /> {running ? "Scoring three ways" : "Run the silence test"}
          </PrimaryButton>
        </>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}

      {result && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-3">
          <ThreeColumns result={result} />
          {verdict && <Finding {...verdict} />}
          <Disclosure>
            <dl className="grid grid-cols-3 gap-2 text-xs">
              {LEGS.map((leg) => {
                const variant = result.variants[leg.key];
                return (
                  <div key={leg.key}>
                    <dt className="text-slate-400">{leg.label}</dt>
                    <dd className="font-mono text-white">
                      {variant.applicable && variant.spoof_probability !== null
                        ? formatScore(variant.spoof_probability)
                        : "n/a"}
                    </dd>
                    <dd className="font-mono text-slate-500">{variant.seconds.toFixed(2)}s</dd>
                  </div>
                );
              })}
            </dl>
            {LEGS.map((leg) =>
              result.variants[leg.key].applicable ? null : (
                <p key={leg.key} className="text-xs text-slate-400">
                  <span className="font-semibold text-slate-200">{leg.label}:</span> {result.variants[leg.key].reason}
                </p>
              ),
            )}
            <div>
              <div className="mb-1 flex justify-between text-xs text-slate-400">
                <span>speech vs non-speech</span>
                <span className="font-mono">
                  {result.speech_seconds.toFixed(2)}s / {result.non_speech_seconds.toFixed(2)}s
                </span>
              </div>
              <div className="flex h-2 overflow-hidden rounded-full bg-white/10">
                <motion.div
                  className="h-full"
                  style={{ background: REAL }}
                  initial={{ width: 0 }}
                  animate={{ width: `${(1 - result.non_speech_fraction) * 100}%` }}
                  transition={{ duration: 0.8 }}
                />
              </div>
            </div>
            <p className="text-xs text-slate-400">
              Silence = audio more than {result.silence_top_db} dB below this clip&apos;s own peak (relative, not an
              absolute noise floor). A silence-only score needs at least {result.min_non_speech_seconds.toFixed(2)}s
              of it. Threshold {result.threshold.toFixed(2)}
              {result.threshold_calibrated ? "" : " (uncalibrated)"}.{result.cached ? " Cached." : ""}
            </p>
          </Disclosure>
        </motion.div>
      )}
    </div>
  );
};

const HEIGHT = 130;

/** Three score columns against the threshold line, growing in turn. */
const ThreeColumns = ({ result }: { result: SilenceProbeResult }) => (
  <div className="relative rounded-2xl bg-white/[0.03] p-3 ring-1 ring-white/10">
    <div className="relative grid grid-cols-3 gap-3" style={{ height: HEIGHT }}>
      {/* threshold line across all three */}
      <div
        className="pointer-events-none absolute inset-x-0 z-10 border-t border-dashed border-white/60"
        style={{ bottom: result.threshold * (HEIGHT - 18) }}
      >
        <span className="absolute -top-4 right-0 text-[10px] text-slate-300">cut {result.threshold.toFixed(2)}</span>
      </div>
      {LEGS.map((leg, index) => (
        <Column key={leg.key} variant={result.variants[leg.key]} delay={index * 0.25} />
      ))}
    </div>
    <div className="mt-2 grid grid-cols-3 gap-3 text-center">
      {LEGS.map((leg) => (
        <div key={leg.key}>
          <div className="text-xs font-semibold text-white">{leg.label}</div>
          <div className="text-[10px] text-slate-400">{leg.hint}</div>
        </div>
      ))}
    </div>
  </div>
);

const Column = ({ variant, delay }: { variant: ProbeVariant; delay: number }) => {
  const { scoreColor } = usePalette();
  if (!variant.applicable || variant.spoof_probability === null) {
    return (
      <div className="flex items-end justify-center">
        <div className="w-full rounded-xl border border-dashed border-white/20 py-3 text-center text-[11px] text-slate-400">
          too little audio
        </div>
      </div>
    );
  }
  const score = variant.spoof_probability;
  const colour = scoreColor(score);
  return (
    <div className="flex flex-col items-center justify-end">
      <motion.span
        className="mb-1 font-mono text-xs font-semibold"
        style={{ color: colour }}
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: delay + 0.5 }}
      >
        {formatScore(score, 2)}
      </motion.span>
      <motion.div
        className="w-full rounded-t-xl"
        style={{ background: `linear-gradient(180deg, ${colour}, ${colour}33)`, boxShadow: `0 0 24px -6px ${colour}` }}
        initial={{ height: 0 }}
        animate={{ height: Math.max(4, score * (HEIGHT - 18)) }}
        transition={{ type: "spring", stiffness: 90, damping: 14, delay }}
      />
    </div>
  );
};

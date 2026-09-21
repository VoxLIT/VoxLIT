import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Bot, Dices, Ear, Sparkles, UserRound } from "lucide-react";
import type { DeepfakeResult, RecordingInfo } from "../types";
import { audioUrlFor, errorMessage, formatBytes, formatSeconds, postDeepfake } from "./api";
import { leanWords } from "./palette";
import { usePalette } from "./theme";
import { ScoreDial } from "./ScoreDial";
import { Disclosure, ErrorNote, PrimaryButton, VerdictChip } from "./ui";
import { WaveformPlayer } from "./WaveformPlayer";

type Guess = "real" | "fake" | null;

interface VerdictPanelProps {
  model: string;
  modelLabel: string;
  recording: RecordingInfo | null;
  onSurprise: () => void;
  surpriseDisabled: boolean;
}

/**
 * The first thing a visitor reads: one clip, a guess, and the detector's
 * answer on a dial. The numbers behind the answer sit one click deeper.
 *
 * The guess is the dataset's "judge before you see the score" exercise turned
 * into a game — it is compared with the DETECTOR, never with the hidden
 * bona fide/spoof label, which this page cannot see.
 */
export const VerdictPanel = ({ model, modelLabel, recording, onSurprise, surpriseDisabled }: VerdictPanelProps) => {
  const { REAL, FAKE, ink } = usePalette();
  const [guess, setGuess] = useState<Guess>(null);
  const [result, setResult] = useState<DeepfakeResult | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tally, setTally] = useState({ asked: 0, agreed: 0 });

  // Each request gets a number; a reply only lands if it is still the latest,
  // so switching clip or model mid-request cannot show the old verdict.
  const latestRequest = useRef(0);

  const recordingId = recording?.recording_id;
  useEffect(() => {
    latestRequest.current += 1;
    setGuess(null);
    setResult(null);
    setError(null);
    setRunning(false);
  }, [recordingId, model]);

  const ask = async () => {
    if (!recordingId) return;
    const request = ++latestRequest.current;
    setRunning(true);
    setError(null);
    try {
      const payload = await postDeepfake<DeepfakeResult>("run", { model, recording_id: recordingId });
      if (request !== latestRequest.current) return;
      setResult(payload);
      if (guess) {
        const agreed = (guess === "fake") === (payload.decision === "spoof");
        setTally((current) => ({ asked: current.asked + 1, agreed: current.agreed + (agreed ? 1 : 0) }));
      }
    } catch (caught) {
      if (request === latestRequest.current) setError(errorMessage(caught, "Detection failed."));
    } finally {
      if (request === latestRequest.current) setRunning(false);
    }
  };

  if (!recording) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-4 p-6 text-center">
        <motion.div
          animate={{ rotate: [0, -8, 8, 0], y: [0, -4, 0] }}
          transition={{ duration: 3, repeat: Infinity, ease: "easeInOut" }}
          className="grid h-16 w-16 place-items-center rounded-2xl bg-white/5 ring-1 ring-white/10"
        >
          <Ear className="h-8 w-8 text-cyan-200" />
        </motion.div>
        <div>
          <div className="font-display text-xl font-semibold text-white">Pick a voice</div>
          <p className="mt-1 text-sm text-slate-400">
            Click a star on the voice map, choose a clip from the library below, or let chance decide.
          </p>
        </div>
        <PrimaryButton onClick={onSurprise} disabled={surpriseDisabled}>
          <Dices className="h-4 w-4" /> Surprise me
        </PrimaryButton>
      </div>
    );
  }

  const spoof = result?.decision === "spoof";
  const agreed = result && guess ? (guess === "fake") === spoof : null;

  return (
    <div className="flex h-full flex-col gap-4 p-5">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[11px] uppercase tracking-widest text-slate-400">Now listening to</div>
          <div className="truncate font-mono text-base font-semibold text-white" title={recording.display_filename}>
            {recording.display_filename}
          </div>
        </div>
        <button
          type="button"
          onClick={onSurprise}
          disabled={surpriseDisabled}
          className="rounded-full p-2 text-slate-400 transition hover:bg-white/10 hover:text-white"
          aria-label="Pick another random clip"
          title="Pick another random clip"
        >
          <Dices className="h-4 w-4" />
        </button>
      </div>

      <WaveformPlayer url={audioUrlFor(recording.recording_id)} label={recording.display_filename} />

      {/* step 1: guess */}
      <div>
        <div className="mb-2 text-sm text-slate-300">
          <span className="font-semibold text-white">Your ears first:</span> real person or machine?
        </div>
        <div className="grid grid-cols-2 gap-2" role="radiogroup" aria-label="Your guess">
          {(["real", "fake"] as const).map((option) => {
            const active = guess === option;
            const color = option === "real" ? REAL : FAKE;
            return (
              <motion.button
                key={option}
                type="button"
                role="radio"
                aria-checked={active}
                disabled={!!result}
                onClick={() => setGuess(option)}
                whileTap={{ scale: 0.95 }}
                animate={{ scale: active ? 1.03 : 1 }}
                className="flex items-center justify-center gap-2 rounded-xl border px-3 py-2.5 text-sm font-semibold transition-colors disabled:cursor-default"
                style={{
                  borderColor: active ? color : ink(0.1),
                  background: active ? `${color}22` : ink(0.03),
                  color: active ? color : ink(0.75),
                  boxShadow: active ? `0 0 24px -6px ${color}` : undefined,
                }}
              >
                {option === "real" ? <UserRound className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                {option === "real" ? "Real person" : "Machine"}
              </motion.button>
            );
          })}
        </div>
      </div>

      {/* step 2: ask */}
      {!result && (
        <PrimaryButton onClick={ask} busy={running} className="w-full">
          <Sparkles className="h-4 w-4" />
          {running ? "The detector is listening…" : guess ? "Reveal the detector's answer" : "Ask the detector"}
        </PrimaryButton>
      )}
      {running && (
        <p className="-mt-2 text-center text-xs text-slate-400">
          The first run for a model downloads it, which can take a few minutes.
        </p>
      )}
      {error && <ErrorNote>{error}</ErrorNote>}

      <AnimatePresence mode="wait">
        {result && (
          <motion.div
            key={result.recording_id + result.model}
            initial={{ opacity: 0, y: 20, filter: "blur(8px)" }}
            animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.5, ease: [0.2, 0.7, 0.3, 1] }}
            className="space-y-3"
          >
            <ScoreDial score={result.spoof_probability} threshold={result.threshold} />

            <div className="text-center">
              <VerdictChip spoof={spoof} />
              <p className="mt-2 text-sm text-slate-300">
                {modelLabel} {leanWords(result.spoof_probability, result.threshold)} towards{" "}
                <span className="font-semibold" style={{ color: spoof ? FAKE : REAL }}>
                  {spoof ? "a synthetic voice" : "a real recording"}
                </span>
                .
              </p>
              {agreed !== null && (
                <motion.p
                  initial={{ scale: 0.6, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  transition={{ type: "spring", stiffness: 260, damping: 14, delay: 0.5 }}
                  className={`mt-2 text-sm font-semibold ${agreed ? "text-emerald-300" : "text-amber-300"}`}
                >
                  {agreed ? "You and the detector agree." : "You and the detector disagree — listen again?"}
                </motion.p>
              )}
              {tally.asked > 0 && (
                <p className="mt-1 text-xs text-slate-500">
                  Agreed on {tally.agreed} of {tally.asked} guesses so far. Neither of you sees the dataset&apos;s answer
                  key.
                </p>
              )}
            </div>

            <Disclosure>
              <dl className="grid grid-cols-2 gap-x-4 gap-y-2">
                <Detail label="spoof score" value={result.spoof_probability.toFixed(3)} />
                <Detail label="bona fide score" value={result.bonafide_probability.toFixed(3)} />
                <Detail label="threshold" value={`${result.threshold.toFixed(2)}${result.threshold_calibrated ? "" : " (uncalibrated)"}`} />
                <Detail label="logits" value={`[${result.logits.map((value) => value.toFixed(2)).join(", ")}]`} />
                <Detail label="spoof class" value={`${result.spoof_index} · ${result.id2label[String(result.spoof_index)]}`} />
                <Detail label="duration" value={formatSeconds(result.duration)} />
                <Detail label="analysed" value={`${formatSeconds(result.analysed_seconds)} of ${formatSeconds(result.analysis_window_seconds)} window`} />
                <Detail label="file size" value={formatBytes(recording.size_bytes)} />
              </dl>
              <p className="text-xs text-slate-400">
                The score is a ranking, not a probability: 0.90 does not mean &ldquo;90% likely fake&rdquo;.
                {!result.threshold_calibrated &&
                  ` ${result.threshold.toFixed(2)} is the naive midpoint, not an equal-error-rate operating point — the Detector report below shows where a calibrated cut would sit.`}
                {result.truncated && ` Only the first ${result.analysed_seconds}s of this ${result.duration}s clip were scored.`}
              </p>
              <p className="break-all font-mono text-[11px] text-slate-500">
                {result.model_id} · {result.threshold_version}
                {result.cached ? " · cached" : ""}
              </p>
            </Disclosure>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const Detail = ({ label, value }: { label: string; value: string }) => (
  <div>
    <dt className="text-[11px] text-slate-400">{label}</dt>
    <dd className="font-mono text-xs text-white">{value}</dd>
  </div>
);

import { RefObject, useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { ExternalLink, Headphones, Loader2, Merge, Minus, Plus, Scissors, Split, Volume2, Wand2, Zap } from "lucide-react";
import { RangeSlider } from "@/components/ui/range-slider";
import { Slider } from "@/components/ui/slider";
import type { DiarizationResult, PerturbationResult, PerturbationType } from "../types";
import { CompareTimelines } from "./CompareTimelines";
import { formatClock } from "./segmentWords";
import { usePalette } from "./theme";
import { Disclosure, ErrorNote, Finding, PrimaryButton } from "./ui";
import { changeBadges, formatShare, perturbationFinding, type BadgeKind } from "./whatIfWords";

interface WhatIfProps {
  result: DiarizationResult;
  perturbationType: PerturbationType;
  setPerturbationType: (type: PerturbationType) => void;
  noisePercent: number;
  setNoisePercent: (value: number) => void;
  maskRange: [number, number];
  setMaskRange: (value: [number, number]) => void;
  perturbation: PerturbationResult | null;
  isPerturbing: boolean;
  perturbationError: string | null;
  runPerturbation: () => void;
  isPreviewing: boolean;
  /** Builds the changed clip without diarizing it, so it can be heard first. */
  previewPerturbation: () => void;
  /** False while the original diarization is still running. */
  canRun: boolean;
  selectedId: string | null;
  onSelectOriginal: (segmentId: string) => void;
  onSelectPerturbed: (segmentId: string) => void;
  perturbedAudioRef: RefObject<HTMLAudioElement>;
  perturbedAudioUrl: string | undefined;
}

const CHOICES: { type: PerturbationType; title: string; text: string }[] = [
  { type: "noise", title: "Add background noise", text: "Hiss over the whole meeting, from a little to a lot." },
  { type: "time_masking", title: "Cut out a chunk", text: "Silence part of the meeting and see who goes missing." },
];

const BADGE_ICON: Record<BadgeKind, typeof Merge> = { merged: Merge, split: Split, disappeared: Minus, appeared: Plus };

/**
 * Step 5: what if the audio was worse? Change the recording one way, run the
 * same model again, and compare the two answers. The comparison is against
 * the original run, not against a human-labelled answer, and says so.
 */
export const WhatIf = ({
  result,
  perturbationType,
  setPerturbationType,
  noisePercent,
  setNoisePercent,
  maskRange,
  setMaskRange,
  perturbation,
  isPerturbing,
  perturbationError,
  runPerturbation,
  isPreviewing,
  previewPerturbation,
  canRun,
  selectedId,
  onSelectOriginal,
  onSelectPerturbed,
  perturbedAudioRef,
  perturbedAudioUrl,
}: WhatIfProps) => {
  const { WARN } = usePalette();

  // Segment ids restart per run, so `seg_3` exists in both. Remember which
  // timeline the last click came from, so only that one lights up.
  const [perturbedPick, setPerturbedPick] = useState<string | null>(null);
  useEffect(() => {
    if (selectedId !== perturbedPick) setPerturbedPick(null);
  }, [selectedId, perturbedPick]);
  useEffect(() => setPerturbedPick(null), [perturbation]);

  // Settings stay put while either request is out: changing them would
  // orphan the clip or run it was started for.
  const busy = isPerturbing || isPreviewing;

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-2" role="radiogroup" aria-label="How to change the audio">
        {CHOICES.map((choice) => {
          const active = choice.type === perturbationType;
          return (
            <motion.button
              key={choice.type}
              type="button"
              role="radio"
              aria-checked={active}
              disabled={busy}
              onClick={() => setPerturbationType(choice.type)}
              whileHover={busy ? undefined : { y: -3 }}
              transition={{ type: "spring", stiffness: 300, damping: 20 }}
              className={`dz-card overflow-hidden rounded-2xl text-left transition-shadow disabled:cursor-not-allowed ${
                active ? "dz-selected" : "opacity-80 hover:opacity-100"
              }`}
            >
              <div className="dz-well h-28 border-b border-white/10">
                {choice.type === "noise" ? <NoiseArt /> : <CutArt />}
              </div>
              <div className="flex items-start gap-3 p-4">
                {choice.type === "noise" ? (
                  <Volume2 className="mt-0.5 h-5 w-5 shrink-0 dz-accent-text" />
                ) : (
                  <Scissors className="mt-0.5 h-5 w-5 shrink-0 dz-accent-text" />
                )}
                <div>
                  <div className="font-display text-lg font-semibold text-white">{choice.title}</div>
                  <div className="text-sm text-slate-400">{choice.text}</div>
                </div>
              </div>
            </motion.button>
          );
        })}
      </div>

      {/* One change at a time: the comparison is two-way, so stacking changes
          would make it impossible to say which one moved the answer. */}
      <div className="dz-card rounded-2xl p-5">
        <AnimatePresence mode="wait" initial={false}>
          {perturbationType === "noise" ? (
            <motion.div key="noise" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
              <div className="mb-3 text-sm font-semibold text-white">How much noise?</div>
              <Slider
                value={[noisePercent]}
                onValueChange={([value]) => setNoisePercent(value)}
                min={1}
                max={100}
                step={1}
                disabled={busy}
                aria-label="Noise amount"
              />
              <div className="mt-2 flex justify-between text-xs text-slate-400">
                <span>a little</span>
                <span>a lot</span>
              </div>
            </motion.div>
          ) : (
            <motion.div key="mask" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -8 }}>
              <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
                <span className="text-sm font-semibold text-white">Which part to cut?</span>
                <span className="font-mono text-xs text-slate-300">
                  {formatClock((maskRange[0] / 100) * result.duration)} – {formatClock((maskRange[1] / 100) * result.duration)}
                </span>
              </div>
              <MiniTimeline result={result} maskRange={maskRange} />
              <div className="mt-3">
                <RangeSlider
                  value={maskRange}
                  onValueChange={setMaskRange}
                  min={0}
                  max={100}
                  step={1}
                  showLabels={false}
                  disabled={busy}
                />
              </div>
              <p className="mt-2 text-xs text-slate-400">Cut out someone&apos;s only turn and they should vanish from the answer.</p>
            </motion.div>
          )}
        </AnimatePresence>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={previewPerturbation}
            disabled={busy}
            className="inline-flex items-center justify-center gap-2 rounded-full bg-white/5 px-5 py-2.5 text-sm font-semibold text-slate-200 ring-1 ring-white/10 transition hover:bg-white/10 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {isPreviewing ? <Loader2 className="h-4 w-4 animate-spin" /> : <Headphones className="h-4 w-4" />}
            {isPreviewing ? "Changing the audio…" : "Hear the changed audio"}
          </button>
          <PrimaryButton onClick={runPerturbation} disabled={!canRun || isPreviewing} busy={isPerturbing}>
            <Wand2 className="h-4 w-4" />
            {isPerturbing ? "Listening again to the changed audio…" : "See what changes"}
          </PrimaryButton>
          <span className="text-xs text-slate-400">
            {!canRun && !isPerturbing
              ? "Waiting for the current run to finish."
              : "A full meeting takes minutes; the same change again is instant."}
          </span>
        </div>
      </div>

      {perturbationError && <ErrorNote>{perturbationError}</ErrorNote>}

      {/* One player for the changed clip, shown as soon as it exists (from a
          preview or a run -- same file). Stays mounted while there is a
          result: clicking an "after" segment seeks this element. */}
      {perturbedAudioUrl && (
        <div className="dz-card rounded-2xl p-4">
          <div className="mb-2 text-sm text-slate-300">
            {perturbation ? "Hear what the second run heard" : "Hear the changed audio"}
          </div>
          <audio ref={perturbedAudioRef} controls className="w-full" src={perturbedAudioUrl} />
        </div>
      )}

      <AnimatePresence>
        {perturbation && (
          <motion.div
            key={perturbation.perturbed.recording_id}
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            className="space-y-4"
          >
            <WhatIfResult
              perturbation={perturbation}
              speakers={result.speakers}
              selectedOriginalId={perturbedPick === null ? selectedId : null}
              selectedPerturbedId={perturbedPick}
              onSelectOriginal={(id) => {
                setPerturbedPick(null);
                onSelectOriginal(id);
              }}
              onSelectPerturbed={(id) => {
                setPerturbedPick(id);
                onSelectPerturbed(id);
              }}
              warn={WARN}
            />

            <WhatIfDetails perturbation={perturbation} perturbedAudioUrl={perturbedAudioUrl} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

const WhatIfResult = ({
  perturbation,
  speakers,
  selectedOriginalId,
  selectedPerturbedId,
  onSelectOriginal,
  onSelectPerturbed,
  warn,
}: {
  perturbation: PerturbationResult;
  speakers: string[];
  selectedOriginalId: string | null;
  selectedPerturbedId: string | null;
  onSelectOriginal: (id: string) => void;
  onSelectPerturbed: (id: string) => void;
  warn: string;
}) => {
  const { delta, original, perturbed } = perturbation;
  const finding = perturbationFinding(delta, original, perturbed, perturbation.perturbation);
  const badges = changeBadges(delta);

  return (
    <>
      <Finding alarming={finding.alarming} title={finding.title} detail={finding.detail} />

      <ul className="flex flex-wrap gap-2" aria-label="Speaker changes">
        {badges.length === 0 ? (
          <li className="rounded-full bg-white/5 px-3 py-1.5 text-xs font-semibold text-slate-300 ring-1 ring-white/10">
            No voice was lost, added, merged or split
          </li>
        ) : (
          badges.map((badge, index) => {
            const Icon = BADGE_ICON[badge.kind];
            return (
              <motion.li
                key={badge.key}
                data-testid="change-badge"
                initial={{ opacity: 0, scale: 0.6, y: 8 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 420, damping: 22, delay: 0.25 + index * 0.08 }}
                className="inline-flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold text-white"
                style={{ background: `color-mix(in srgb, ${warn} 18%, transparent)`, boxShadow: `inset 0 0 0 1px ${warn}` }}
              >
                <Icon className="h-3.5 w-3.5" style={{ color: warn }} />
                {badge.text}
              </motion.li>
            );
          })
        )}
      </ul>

      <CompareTimelines
        original={original}
        perturbed={perturbed}
        delta={delta}
        speakers={speakers}
        selectedOriginalId={selectedOriginalId}
        selectedPerturbedId={selectedPerturbedId}
        onSelectOriginal={onSelectOriginal}
        onSelectPerturbed={onSelectPerturbed}
      />
    </>
  );
};

/** Level 2. The headline number is labelled for what it is: how far the
 *  answer moved from the original run. The original run is the reference
 *  here, not a human-labelled answer, so it cannot say which run is right. */
const WhatIfDetails = ({
  perturbation,
  perturbedAudioUrl,
}: {
  perturbation: PerturbationResult;
  perturbedAudioUrl: string | undefined;
}) => {
  const { delta } = perturbation;
  const { der_components: components } = delta;
  const reference = components.total || 1;
  const bars = [
    { label: "Missed (speech the second run did not pick up)", value: components.missed_detection },
    { label: "False alarm (speech the second run added)", value: components.false_alarm },
    { label: "Confusion (speech given to a different speaker)", value: components.confusion },
  ];
  const seed = perturbation.perturbation.params.seed;
  const mapping = Object.entries(delta.speaker_mapping);

  return (
    <Disclosure>
      <div>
        <div className="text-xs uppercase tracking-wider text-slate-400">Change from the original run</div>
        <div className="mt-1 flex flex-wrap items-baseline gap-3">
          <span className="font-mono text-3xl font-semibold text-white" data-testid="der-value">
            {(delta.der * 100).toFixed(1)}%
          </span>
          {perturbation.cached && (
            <span className="inline-flex items-center gap-1 rounded-full bg-white/5 px-2 py-0.5 text-xs text-slate-300 ring-1 ring-white/10">
              <Zap className="h-3 w-3 dz-accent-text" /> cached
            </span>
          )}
        </div>
        <p className="mt-1 text-xs text-slate-400">
          Diarization error rate (DER) with the <em>original run</em> as the reference, not a human-labelled answer. It
          measures how far the answer moved, not which run is right.
        </p>
      </div>

      <div className="space-y-2">
        <div className="text-xs text-slate-400">What moved, in seconds of the original run&apos;s speech ({components.total} s)</div>
        {bars.map((bar, index) => (
          <div key={bar.label}>
            <div className="flex justify-between text-xs">
              <span className="text-slate-300">{bar.label}</span>
              <span className="font-mono text-slate-200">
                {bar.value} s · {formatShare(bar.value / reference)}
              </span>
            </div>
            <div className="mt-1 h-2 overflow-hidden rounded-full bg-white/[0.06]">
              <motion.div
                className="h-full rounded-full dz-accent-bar"
                initial={{ width: 0 }}
                animate={{ width: `${Math.min(100, (bar.value / reference) * 100)}%` }}
                transition={{ duration: 0.8, delay: 0.1 + index * 0.1, ease: [0.2, 0.7, 0.3, 1] }}
              />
            </div>
          </div>
        ))}
      </div>

      <div>
        <div className="text-xs text-slate-400">
          Boundaries that moved more than {delta.boundary_shifts.threshold_seconds} s: {delta.boundary_shifts.count}
        </div>
        {delta.boundary_shifts.shifts.length > 0 && (
          <ul className="mt-1 max-h-40 space-y-0.5 overflow-y-auto font-mono text-xs">
            {delta.boundary_shifts.shifts.map((shift) => (
              <li key={`${shift.segment_id}-${shift.edge}`}>
                {shift.segment_id} {shift.edge}: {formatClock(shift.original)} → {formatClock(shift.perturbed)} (
                {shift.delta > 0 ? "+" : ""}
                {shift.delta.toFixed(2)} s)
              </li>
            ))}
          </ul>
        )}
      </div>

      <dl className="grid gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
        <div>
          <dt className="inline text-slate-400">Change: </dt>
          <dd className="inline font-mono">
            {perturbation.perturbation.type}{" "}
            {Object.entries(perturbation.perturbation.params)
              .filter(([key]) => key !== "seed")
              .map(([key, value]) => `${key}=${value}`)
              .join(", ")}
          </dd>
        </div>
        <div>
          <dt className="inline text-slate-400">Noise seed: </dt>
          <dd className="inline font-mono">{seed !== undefined ? seed : "not used (cutting is deterministic)"}</dd>
        </div>
        <div className="sm:col-span-2">
          <dt className="inline text-slate-400">Label alignment (Hungarian matching, after → before): </dt>
          <dd className="inline font-mono">
            {mapping.length === 0 ? "no speaker matched" : mapping.map(([after, before]) => `${after}→${before}`).join(", ")}
          </dd>
        </div>
        <div>
          <dt className="inline text-slate-400">Segments with no counterpart: </dt>
          <dd className="inline font-mono">{delta.lost_segments.length}</dd>
        </div>
      </dl>

      {perturbedAudioUrl && (
        <a
          href={perturbedAudioUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1.5 text-xs font-semibold dz-accent-text hover:underline"
        >
          <ExternalLink className="h-3.5 w-3.5" /> Open the changed audio
        </a>
      )}
    </Disclosure>
  );
};

/** The original run as one thin strip, with the part that will be cut shaded. */
const MiniTimeline = ({ result, maskRange }: { result: DiarizationResult; maskRange: [number, number] }) => {
  const { speakerColor } = usePalette();
  const duration = result.duration || 1;
  return (
    <div className="dz-well relative h-6 overflow-hidden rounded-md" aria-hidden>
      {result.segments.map((segment) => (
        <div
          key={segment.id}
          className="absolute top-1 bottom-1 rounded-sm"
          style={{
            left: `${(segment.start / duration) * 100}%`,
            width: `${Math.max(((segment.end - segment.start) / duration) * 100, 0.15)}%`,
            background: speakerColor(result.speakers, segment.speaker),
          }}
        />
      ))}
      <motion.div
        className="dz-hatch absolute inset-y-0 border-x-2 text-slate-400"
        animate={{ left: `${maskRange[0]}%`, width: `${maskRange[1] - maskRange[0]}%` }}
        transition={{ type: "spring", stiffness: 400, damping: 35 }}
        style={{ background: "color-mix(in srgb, var(--dz-canvas) 70%, transparent)", borderColor: "var(--dz-accent)" }}
      />
    </div>
  );
};

// --- card pictures: stand-ins until Step 11 adds credited photos -----------------

const BARS = Array.from({ length: 48 }, (_, i) => 0.25 + 0.7 * Math.abs(Math.sin(i * 0.9) * Math.cos(i * 0.37)));

const NoiseArt = () => {
  const { ACCENT, ink } = usePalette();
  return (
    <svg viewBox="0 0 480 112" preserveAspectRatio="none" className="h-full w-full" aria-hidden>
      {Array.from({ length: 160 }, (_, i) => (
        <circle key={i} cx={(i * 97) % 480} cy={(i * 53) % 112} r={1.2} fill={ink(0.25)} />
      ))}
      {BARS.map((h, i) => (
        <rect key={i} x={12 + i * 9.6} y={56 - h * 40} width={5} height={h * 80} rx={2.5} fill={ACCENT} opacity={0.85} />
      ))}
    </svg>
  );
};

const CutArt = () => {
  const { ACCENT, ink } = usePalette();
  return (
    <svg viewBox="0 0 480 112" preserveAspectRatio="none" className="h-full w-full" aria-hidden>
      {BARS.map((h, i) => {
        const cut = i >= 18 && i < 30;
        return (
          <rect
            key={i}
            x={12 + i * 9.6}
            y={cut ? 55 : 56 - h * 40}
            width={5}
            height={cut ? 2 : h * 80}
            rx={2.5}
            fill={cut ? ink(0.3) : ACCENT}
            opacity={0.85}
          />
        );
      })}
      <rect x={182} y={8} width={115} height={96} rx={8} fill="none" stroke={ink(0.45)} strokeDasharray="5 5" />
    </svg>
  );
};

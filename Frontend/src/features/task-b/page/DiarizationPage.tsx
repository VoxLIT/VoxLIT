import { ReactNode, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { motion, useScroll, useSpring } from "motion/react";
import { ArrowDown, AudioWaveform, Ear, Layers, Scissors, Sparkles, Users, Zap } from "lucide-react";
import type { TaskDefinition } from "@/tasks/types";
import { DiarizationTimeline } from "../DiarizationTimeline";
import { HeroConversation } from "./HeroConversation";
import { MeetingLibrary } from "./MeetingLibrary";
import { RunProgress } from "./RunProgress";
import { SegmentMap } from "./SegmentMap";
import { SimilarityView } from "./SimilarityView";
import { WhatIf } from "./WhatIf";
import { Disclosure, ErrorNote, PrimaryButton, SectionTitle } from "./ui";
import { useDiarization } from "./useDiarization";
import { DzThemeProvider, ThemeToggle, usePalette, type DzTheme } from "./theme";
import "./diarization-page.css";

const NAV = [
  { href: "#meetings", label: "Pick a meeting" },
  { href: "#results", label: "Results" },
  { href: "#map", label: "Segment map" },
  { href: "#hesitate", label: "Hesitation" },
  { href: "#what-if", label: "What if" },
];

/**
 * Speaker Diarization's own page. One long scroll: "Who spoke when?", the
 * meeting library, then the results. Technical detail sits one click deeper,
 * in "Technical details" drawers.
 *
 * The results section still mounts the pre-redesign speaker timeline until
 * Step 7 replaces it (guide Part D).
 */
export const DiarizationPage = ({ task }: { task: TaskDefinition }) => (
  <DzThemeProvider>{(theme) => <DiarizationPageContent task={task} theme={theme} />}</DzThemeProvider>
);

const DiarizationPageContent = ({ task, theme }: { task: TaskDefinition; theme: DzTheme }) => {
  const { CANVAS } = usePalette();
  const availableModels = task.models.filter((option) => option.available);
  const [model, setModel] = useState(task.defaultModel ?? availableModels[0]?.id ?? "");
  const modelLabel = task.models.find((option) => option.id === model)?.label ?? model;
  const d = useDiarization(model);
  // Switching model mid-run would clear the screen while the old model's
  // /run is still in flight, and its answer would then land under the new name.
  const busy = d.isRunning || d.isPerturbing;

  // Overscroll and the moments before the page paints match the theme too.
  useEffect(() => {
    const previous = document.body.style.background;
    document.body.style.background = CANVAS;
    return () => {
      document.body.style.background = previous;
    };
  }, [CANVAS]);

  const scrollTo = (id: string) => document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });

  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 30 });

  return (
    <div className="dz-page relative min-h-screen overflow-x-hidden" data-theme={theme}>
      <div className="dz-glow" aria-hidden />
      <motion.div
        className="fixed inset-x-0 top-0 z-50 h-0.5 origin-left dz-accent-bar"
        style={{ scaleX: progress }}
        aria-hidden
      />

      {/* header */}
      <header className="sticky top-0 z-40 dz-header border-b border-white/10 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <Link to="/" className="flex items-center gap-2 font-display text-lg font-bold text-white">
            <AudioWaveform className="h-5 w-5 dz-accent-text" />
            VoxLIT
          </Link>
          <span className="hidden text-sm text-slate-400 md:inline">{task.name}</span>
          <nav className="hidden items-center gap-1 lg:flex" aria-label="Sections">
            {NAV.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="rounded-full px-3 py-1 text-sm text-slate-300 transition hover:bg-white/10 hover:text-white"
              >
                {item.label}
              </a>
            ))}
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-2" role="radiogroup" aria-label="Diarization model">
            <span className="text-xs text-slate-400">Model</span>
            {availableModels.map((option) => {
              const active = option.id === model;
              return (
                <button
                  key={option.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  disabled={busy && !active}
                  title={busy && !active ? "Wait for the current run to finish" : option.label}
                  onClick={() => setModel(option.id)}
                  className="relative rounded-full px-3 py-1.5 text-xs font-semibold text-slate-300 ring-1 ring-white/10 transition-colors hover:text-white disabled:cursor-not-allowed disabled:opacity-40 aria-checked:text-slate-950"
                >
                  {active && (
                    <motion.span
                      layoutId="dz-model-pill"
                      className="absolute inset-0 rounded-full dz-pill"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                  <span className="relative">{option.label}</span>
                </button>
              );
            })}
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="relative z-10">
        <Hero onStart={() => scrollTo("meetings")} />

        {/* Step 1: pick a meeting */}
        <section id="meetings" className="mx-auto max-w-[1600px] scroll-mt-20 px-4 py-12 sm:px-6">
          <SectionTitle eyebrow="Step 1 · Pick a meeting" title={<>Choose a <span className="dz-accent-text">recording</span></>}>
            Real meetings from the AMI corpus, or your own recording. Press ▶ to peek, then select one.
          </SectionTitle>

          <div className="space-y-5">
            {d.error && <ErrorNote>{d.error}</ErrorNote>}

            <MeetingLibrary
              recordings={d.recordings}
              uploads={d.uploads}
              selectedId={d.selectedRecordingId}
              onSelect={d.select}
              onUpload={d.upload}
              isUploading={d.isUploading}
              disabled={busy}
              isLoading={d.isLoadingRecordings}
            />

            <div className="flex flex-wrap items-center gap-3">
              <PrimaryButton
                onClick={d.run}
                disabled={!d.selectedRecordingId || d.isPerturbing}
                busy={d.isRunning}
                className="px-7 py-3 text-base"
              >
                <Users className="h-5 w-5" />
                {d.isRunning ? "Finding the speakers…" : "Find the speakers"}
              </PrimaryButton>
              <span className="text-sm text-slate-400">
                {d.selectedRecordingId ? (
                  <>
                    with <span className="font-semibold text-slate-200">{modelLabel}</span>
                  </>
                ) : (
                  "Select a meeting first."
                )}
              </span>
              {d.result && !d.isRunning && (
                <motion.span
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  className="inline-flex items-center gap-1 rounded-full bg-white/5 px-2.5 py-1 text-xs font-semibold text-slate-300 ring-1 ring-white/10"
                >
                  {d.result.cached ? (
                    <>
                      <Zap className="h-3 w-3 dz-accent-text" /> cached
                    </>
                  ) : (
                    "fresh run"
                  )}
                </motion.span>
              )}
            </div>

            <RunProgress running={d.isRunning} />

            {d.glassBoxNote && (
              <Disclosure title={<>About {modelLabel}</>}>
                <p>{d.glassBoxNote}</p>
              </Disclosure>
            )}

            {/* Interim player: seeking and pair playback in the views below
                drive this element. Step 7 replaces it with a custom player. */}
            {d.audioUrl && (
              <audio
                ref={d.audioRef}
                controls
                className="w-full"
                src={d.audioUrl}
                onTimeUpdate={d.handleTimeUpdate}
              />
            )}
          </div>
        </section>

        {/* Results. The speaker timeline is still the pre-redesign view until Step 7. */}
        <section id="results" className="mx-auto max-w-[1600px] scroll-mt-20 space-y-4 px-4 py-12 sm:px-6">
          {d.result ? (
            <>
              <SectionTitle eyebrow="Results" title="Who spoke when">
                {d.result.num_speakers} speakers, {d.result.segments.length} segments.
              </SectionTitle>

              <LegacyPanel title="Speaker timeline">
                <DiarizationTimeline
                  segments={d.result.segments}
                  speakers={d.result.speakers}
                  duration={d.result.duration}
                  hoveredId={d.hoveredId}
                  selectedId={d.selectedId}
                  onHover={d.setHoveredId}
                  onSelect={d.seekToSegment}
                />
              </LegacyPanel>

              {/* Step 3: never gated on the projection — its side columns work without it. */}
              <div id="map" className="scroll-mt-20 pt-8">
                <SectionTitle
                  eyebrow="Step 3 · Which voices sound alike?"
                  title={<>The <span className="dz-accent-text">segment map</span></>}
                >
                  Every dot is a moment of speech. Dots that sound alike sit close together — pick one to see its closest
                  matches.
                </SectionTitle>
                <SegmentMap
                  result={d.result}
                  projection={d.projection}
                  projectionError={d.projectionError}
                  dims={d.mapDims}
                  onDimsChange={d.setMapDims}
                  projection3d={d.projection3d}
                  projection3dError={d.projection3dError}
                  isRunning={d.isRunning}
                  embeddingDimension={d.embeddingDimension}
                  selectedId={d.selectedId}
                  hoveredId={d.hoveredId}
                  onHover={d.setHoveredId}
                  onSelect={d.seekToSegment}
                  onPlay={(id) => d.playPair(id, id)}
                />
              </div>

              {/* Step 4 */}
              <div id="hesitate" className="scroll-mt-20 pt-8">
                <SectionTitle
                  eyebrow="Step 4 · Where did it hesitate?"
                  title={<>Voices it could <span className="dz-accent-text">mix up</span></>}
                >
                  Every moment of speech compared with every other. Bright squares between different speakers are where
                  the model had to make a close call.
                </SectionTitle>
                <SimilarityView
                  result={d.result}
                  selectedId={d.selectedId}
                  onSelect={d.seekToSegment}
                  onPlay={(id) => d.playPair(id, id)}
                  onPlayPair={d.playPair}
                />
              </div>

              {/* Step 5 */}
              <div id="what-if" className="scroll-mt-20 pt-8">
                <SectionTitle
                  eyebrow="Step 5 · What if the audio was worse?"
                  title={<>Make it <span className="dz-accent-text">harder</span></>}
                >
                  Change the recording, run the same model again, and see how the answer moves.
                </SectionTitle>
                <WhatIf
                  result={d.result}
                  perturbationType={d.perturbationType}
                  setPerturbationType={d.setPerturbationType}
                  noisePercent={d.noisePercent}
                  setNoisePercent={d.setNoisePercent}
                  maskRange={d.maskRange}
                  setMaskRange={d.setMaskRange}
                  perturbation={d.perturbation}
                  isPerturbing={d.isPerturbing}
                  perturbationError={d.perturbationError}
                  runPerturbation={d.runPerturbation}
                  isPreviewing={d.isPreviewing}
                  previewPerturbation={d.previewPerturbation}
                  canRun={!!d.selectedRecordingId && !d.isRunning}
                  selectedId={d.selectedId}
                  onSelectOriginal={d.seekToSegment}
                  onSelectPerturbed={d.seekPerturbedSegment}
                  perturbedAudioRef={d.perturbedAudioRef}
                  perturbedAudioUrl={d.perturbedAudioUrl}
                />
              </div>
            </>
          ) : (
            <p className="dz-card rounded-2xl p-6 text-center text-sm text-slate-400">
              Pick a meeting above and press <span className="font-semibold text-slate-200">Find the speakers</span> to
              see who spoke when.
            </p>
          )}
        </section>

        <footer className="mx-auto max-w-[1600px] border-t border-white/10 px-4 py-8 text-xs text-slate-500 sm:px-6">
          Meetings from the AMI Meeting Corpus. Speaker colours follow the run&apos;s own speaker labels.
        </footer>
      </main>
    </div>
  );
};

/** Holder for a pre-redesign view. The views themselves are untouched;
 *  `light` keeps the holder light in both themes for views whose hardcoded
 *  button fills disappear on a dark card. */
const LegacyPanel = ({ title, light = false, children }: { title: string; light?: boolean; children: ReactNode }) => (
  <div className={`dz-legacy rounded-2xl p-4 ${light ? "dz-legacy-light" : ""}`}>
    <h3 className="mb-3 text-sm font-semibold">{title}</h3>
    {children}
  </div>
);

const Hero = ({ onStart }: { onStart: () => void }) => (
  <section className="mx-auto grid max-w-[1600px] items-center gap-10 px-4 pb-6 pt-12 sm:px-6 lg:grid-cols-[1fr_1.1fr] lg:pt-20">
    <div>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4 inline-flex items-center gap-2 rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300 ring-1 ring-white/10"
      >
        <Sparkles className="h-3.5 w-3.5 dz-accent-text" /> Speaker diarization
      </motion.div>
      <motion.h1
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.2, 0.7, 0.3, 1] }}
        className="font-display text-5xl font-bold leading-[1.05] text-white sm:text-6xl"
      >
        Who spoke <span className="dz-accent-text">when?</span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-5 max-w-xl text-lg text-slate-300"
      >
        Drop in a meeting recording and see each voice get its own colour.
      </motion.p>

      <ol className="mt-6 grid gap-3 sm:grid-cols-3">
        {[
          { icon: Ear, title: "Listen", text: "Pick a meeting and play it" },
          { icon: Scissors, title: "Split", text: "The model finds each voice" },
          { icon: Layers, title: "Explore", text: "See why it grouped them" },
        ].map((step, index) => (
          <motion.li
            key={step.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 + index * 0.1 }}
            className="dz-card rounded-2xl p-3"
          >
            <step.icon className="h-5 w-5 dz-accent-text" />
            <div className="mt-2 text-sm font-semibold text-white">
              {index + 1}. {step.title}
            </div>
            <div className="text-xs text-slate-400">{step.text}</div>
          </motion.li>
        ))}
      </ol>

      <div className="mt-7">
        <PrimaryButton onClick={onStart}>
          Pick a meeting <ArrowDown className="h-4 w-4" />
        </PrimaryButton>
      </div>
    </div>

    <motion.figure
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ type: "spring", stiffness: 60, damping: 14 }}
      className="dz-card rounded-3xl p-4 sm:p-6"
    >
      <HeroConversation />
    </motion.figure>
  </section>
);

export default DiarizationPage;

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useScroll, useSpring } from "motion/react";
import { ArrowDown, AudioWaveform, Bot, Dices, Ear, Lock, Mic, Sparkles, UserRound } from "lucide-react";
import { API_BASE } from "@/lib/api";
import type { TaskDefinition } from "@/tasks/types";
import type { DeepfakeEmbeddingProjection, EmbeddingRecording, RecordingInfo, UserClip } from "../types";
import { AllDetectors } from "./AllDetectors";
import { errorMessage, listUserClips, postDeepfake } from "./api";
import { DetectorReport } from "./DetectorReport";
import { ListeningHeatmap } from "./ListeningHeatmap";
import { SilenceTest } from "./SilenceTest";
import { ErrorNote, FeatureImage, PrimaryButton, SectionTitle } from "./ui";
import { HumanVoiceArt, MachineVoiceArt } from "./HeroIllustrations";
import { VerdictPanel } from "./VerdictPanel";
import { VoiceLibrary } from "./VoiceLibrary";
import { VoiceMap, type ReductionMethod } from "./VoiceMap";
import { YourClips } from "./YourClips";
import { DfThemeProvider, ThemeToggle, usePalette, type DfTheme } from "./theme";
import "./deepfake-page.css";

const MODEL_BLURBS: Record<string, string> = {
  "xlsr-deepfake": "listens to raw sound waves",
  "ast-fakeaudio": "reads the spectrogram like an image",
  "xlsr-mamba": "a state-space model over wav2vec2 features",
};

const NAV = [
  { href: "#your-voice", label: "Your voice" },
  { href: "#listen", label: "Listen" },
  { href: "#compare", label: "All detectors" },
  { href: "#report", label: "Detector report" },
  { href: "#library", label: "Library" },
];

/**
 * Audio Deepfake Detection's own page. One long scroll: a playful first look
 * (pick a voice, guess, ask the detector), the voice map in the middle with
 * the clip's verdict and explanations either side, the detector's report
 * card, and the whole dataset laid out at the bottom. Technical detail always
 * sits one click deeper, in "Technical details" drawers.
 */
export const DeepfakePage = ({ task }: { task: TaskDefinition }) => (
  <DfThemeProvider>{(theme) => <DeepfakePageContent task={task} theme={theme} />}</DfThemeProvider>
);

const DeepfakePageContent = ({ task, theme }: { task: TaskDefinition; theme: DfTheme }) => {
  const { CANVAS } = usePalette();
  const availableModels = task.models.filter((option) => option.available);
  const [model, setModel] = useState(task.defaultModel ?? availableModels[0]?.id ?? "");
  const modelLabel = task.models.find((option) => option.id === model)?.label ?? model;

  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState("");
  const [userClips, setUserClips] = useState<UserClip[]>([]);

  const [mapStarted, setMapStarted] = useState(false);
  const [method, setMethod] = useState<ReductionMethod>("pca");
  const [is3D, setIs3D] = useState(false);
  const [refreshToken, setRefreshToken] = useState(0);
  const [projection, setProjection] = useState<DeepfakeEmbeddingProjection | null>(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);

  const stage = useRef<HTMLElement>(null);

  // Overscroll and the moments before the page paints match the theme too.
  useEffect(() => {
    const previous = document.body.style.background;
    document.body.style.background = CANVAS;
    return () => {
      document.body.style.background = previous;
    };
  }, [CANVAS]);

  useEffect(() => {
    const load = async () => {
      try {
        const response = await fetch(`${API_BASE}/tasks/deepfake/dataset/recordings`, { credentials: "include" });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Could not list recordings.");
        setRecordings(payload.recordings as RecordingInfo[]);
      } catch (caught) {
        setListError(errorMessage(caught, "Could not list recordings."));
      }
    };
    load();
    // The visitor's own clips from earlier in this session. Optional: the page
    // works without them, so a failure here stays silent.
    listUserClips()
      .then(setUserClips)
      .catch(() => undefined);
  }, []);

  // Placed on the same map as the dataset, so the visitor can see which
  // dataset clips the detector thinks their voice resembles.
  const userClipIds = useMemo(() => userClips.map((clip) => clip.recording_id), [userClips]);
  const userClipKey = userClipIds.join(",");

  useEffect(() => {
    if (!mapStarted || recordings.length === 0) return;
    const controller = new AbortController();
    setMapLoading(true);
    setMapError(null);
    postDeepfake<DeepfakeEmbeddingProjection>(
      "embeddings",
      {
        model,
        reduction_method: method,
        n_components: is3D ? 3 : 2,
        ...(userClipIds.length ? { extra_recording_ids: userClipIds } : {}),
      },
      controller.signal,
    )
      .then((payload) => {
        setProjection(payload);
        setMapLoading(false);
      })
      .catch((caught) => {
        if (caught instanceof DOMException && caught.name === "AbortError") return;
        setMapError(errorMessage(caught, "The voice map failed."));
        setMapLoading(false);
      });
    return () => controller.abort();
    // userClipIds is covered by userClipKey.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mapStarted, recordings.length, model, method, is3D, refreshToken, userClipKey]);

  // Points from another detector would silently belong to the wrong model.
  const liveProjection = projection && projection.model === model ? projection : null;

  const recordingsById = useMemo(
    () => new Map<string, RecordingInfo>([...recordings, ...userClips].map((recording) => [recording.recording_id, recording])),
    [recordings, userClips],
  );
  const scores = useMemo(
    () =>
      new Map<string, EmbeddingRecording>(
        (liveProjection?.recordings ?? []).map((point) => [point.recording_id, point]),
      ),
    [liveProjection],
  );
  const selected = recordingsById.get(selectedId) ?? null;

  const select = useCallback((recordingId: string, scroll = false) => {
    setSelectedId(recordingId);
    if (scroll) stage.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  const addUserClip = useCallback(
    (clip: UserClip) => {
      setUserClips((current) => [clip, ...current.filter((item) => item.recording_id !== clip.recording_id)]);
      select(clip.recording_id, true);
    },
    [select],
  );
  const removeUserClip = useCallback((recordingId: string) => {
    setUserClips((current) => current.filter((clip) => clip.recording_id !== recordingId));
    setSelectedId((current) => (current === recordingId ? "" : current));
  }, []);

  const surprise = useCallback(() => {
    if (recordings.length === 0) return;
    const pool = recordings.filter((recording) => recording.recording_id !== selectedId);
    const pick = pool[Math.floor(Math.random() * pool.length)] ?? recordings[0];
    select(pick.recording_id, true);
  }, [recordings, selectedId, select]);

  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 30 });

  return (
    <div className="df-page relative min-h-screen overflow-x-hidden" data-theme={theme}>
      <div className="df-aurora" aria-hidden />
      <div className="df-grid" aria-hidden />
      <motion.div
        className="fixed inset-x-0 top-0 z-50 h-0.5 origin-left df-accent-bar"
        style={{ scaleX: progress }}
        aria-hidden
      />

      {/* header */}
      <header className="sticky top-0 z-40 df-header border-b border-white/10 backdrop-blur-xl">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-6 gap-y-2 px-4 py-3 sm:px-6">
          <Link to="/" className="flex items-center gap-2 font-display text-lg font-bold text-white">
            <AudioWaveform className="h-5 w-5 text-cyan-300" />
            VoxLIT
          </Link>
          <span className="hidden text-sm text-slate-400 md:inline">{task.name}</span>
          <nav className="hidden items-center gap-1 lg:flex" aria-label="Sections">
            {NAV.map((item) => (
              <a key={item.href} href={item.href} className="rounded-full px-3 py-1 text-sm text-slate-300 transition hover:bg-white/10 hover:text-white">
                {item.label}
              </a>
            ))}
          </nav>
          <div className="ml-auto flex flex-wrap items-center gap-2" role="radiogroup" aria-label="Detector model">
            <span className="text-xs text-slate-400">Detector</span>
            {availableModels.map((option) => {
              const active = option.id === model;
              return (
                <button
                  key={option.id}
                  type="button"
                  role="radio"
                  aria-checked={active}
                  title={`${option.label} — ${MODEL_BLURBS[option.id] ?? ""}`}
                  onClick={() => setModel(option.id)}
                  className="relative rounded-full px-3 py-1.5 text-xs font-semibold text-slate-300 ring-1 ring-white/10 transition-colors hover:text-white aria-checked:text-slate-950"
                >
                  {active && (
                    <motion.span
                      layoutId="df-model-pill"
                      className="absolute inset-0 rounded-full df-pill"
                      transition={{ type: "spring", stiffness: 380, damping: 30 }}
                    />
                  )}
                  <span className="relative">{option.label.replace(/\s*\(Model [A-Z]\)/, "")}</span>
                </button>
              );
            })}
          </div>
          <ThemeToggle />
        </div>
      </header>

      <main className="relative z-10">
        <Hero
          onStart={() => stage.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
          onOwnVoice={() => document.getElementById("your-voice")?.scrollIntoView({ behavior: "smooth", block: "start" })}
          onSurprise={surprise}
          ready={recordings.length > 0}
          modelLabel={modelLabel}
          blurb={MODEL_BLURBS[model]}
        />

        {listError && (
          <div className="mx-auto max-w-[1600px] px-4 sm:px-6">
            <ErrorNote>{listError}</ErrorNote>
          </div>
        )}

        <section id="your-voice" className="mx-auto max-w-[1600px] scroll-mt-20 px-4 py-12 sm:px-6">
          <SectionTitle eyebrow="Bring your own" title={<>Test <span className="df-gradient-text">your own voice</span></>}>
            Upload a clip or record one right here. It joins the page like any dataset clip: all three detectors, the
            silence test, the heatmap and the voice map all work on it. Nobody knows its answer but you.
          </SectionTitle>
          <YourClips
            clips={userClips}
            selectedId={selectedId}
            onAdded={addUserClip}
            onRemoved={removeUserClip}
            onSelect={(recordingId) => select(recordingId, true)}
          />
        </section>

        {/* the stage: verdict | map | explanations */}
        <section id="listen" ref={stage} className="mx-auto max-w-[1600px] scroll-mt-20 px-4 py-12 sm:px-6">
          <SectionTitle eyebrow="Step 1 · Listen" title={<>Pick a voice. <span className="df-gradient-text">Trust your ears?</span></>}>
            Hover the stars to peek at clips, click one to study it. Its verdict appears on the left and the
            explanations on the right.
          </SectionTitle>

          <div className="grid gap-5 xl:grid-cols-[minmax(300px,360px)_minmax(0,1fr)_minmax(300px,380px)]">
            <motion.aside
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className={`df-glass rounded-3xl ${selected ? "df-glow-border" : ""}`}
              aria-label="Verdict"
            >
              <VerdictPanel
                model={model}
                modelLabel={modelLabel}
                recording={selected}
                onSurprise={surprise}
                surpriseDisabled={recordings.length === 0}
              />
            </motion.aside>

            <motion.div
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true }}
              className="df-glass order-first rounded-3xl xl:order-none"
            >
              <VoiceMap
                modelLabel={modelLabel}
                datasetSize={recordings.length}
                started={mapStarted}
                onStart={() => setMapStarted(true)}
                projection={liveProjection}
                loading={mapLoading}
                error={mapError}
                method={method}
                onMethodChange={setMethod}
                is3D={is3D}
                on3DChange={setIs3D}
                onRefresh={() => setRefreshToken((token) => token + 1)}
                recordingsById={recordingsById}
                selectedId={selectedId}
                onSelect={(recordingId) => select(recordingId)}
              />
            </motion.div>

            <motion.aside
              initial={{ opacity: 0, x: 30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className="df-glass rounded-3xl p-5"
              aria-label="Explain this clip"
            >
              <div className="mb-4">
                <div className="text-[11px] uppercase tracking-widest text-slate-400">Step 2 · Look closer</div>
                <h3 className="font-display text-xl font-semibold text-white">Why does it think so?</h3>
              </div>
              <AnimatePresence mode="wait">
                {selected ? (
                  <motion.div
                    key={selected.recording_id + model}
                    initial={{ opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -12 }}
                    className="space-y-6"
                  >
                    <SilenceTest model={model} recordingId={selected.recording_id} />
                    <div className="h-px bg-gradient-to-r from-transparent via-white/20 to-transparent" />
                    <ListeningHeatmap model={model} recordingId={selected.recording_id} />
                  </motion.div>
                ) : (
                  <motion.div key="locked" initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="space-y-4">
                    {[
                      { src: "/deepfake/silence-probe.png", title: "Voice or silence?", text: "Tests whether the verdict rests on the pauses rather than the voice." },
                      { src: "/deepfake/saliency.png", title: "Where did it listen?", text: "A heatmap of the moments that pushed it towards “synthetic”." },
                    ].map((item) => (
                      <div key={item.title} className="relative">
                        <FeatureImage src={item.src} alt="" className="h-28 opacity-50 grayscale" />
                        <div className="absolute inset-0 flex flex-col justify-end p-3">
                          <div className="flex items-center gap-1.5 font-display font-semibold text-white">
                            <Lock className="h-3.5 w-3.5" /> {item.title}
                          </div>
                          <p className="text-xs text-slate-300">{item.text}</p>
                        </div>
                      </div>
                    ))}
                    <p className="text-center text-sm text-slate-400">Pick a clip to unlock these.</p>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.aside>
          </div>
        </section>

        <section id="compare" className="mx-auto max-w-[1600px] scroll-mt-20 px-4 py-12 sm:px-6">
          <SectionTitle eyebrow="Step 2½ · Second opinions" title={<>Ask <span className="df-gradient-text">all {availableModels.length} detectors</span></>}>
            Every detector hears the same clip differently. Run the verdict, the silence test and the heatmap on each of
            them and compare.
          </SectionTitle>
          {selected ? (
            <AllDetectors models={availableModels} recording={selected} />
          ) : (
            <p className="df-glass rounded-3xl p-6 text-center text-sm text-slate-400">
              Pick a clip above — or upload or record your own — to compare the detectors on it.
            </p>
          )}
        </section>

        <section id="report" className="mx-auto max-w-[1600px] scroll-mt-20 px-4 py-16 sm:px-6">
          <SectionTitle eyebrow="Step 3 · The big picture" title={<>How good is <span className="df-gradient-text">{modelLabel}</span>?</>}>
            A single verdict can be lucky. Test the detector on every labelled clip and see the mistakes it makes.
          </SectionTitle>
          <DetectorReport model={model} modelLabel={modelLabel} datasetSize={recordings.length} />
        </section>

        <section id="library" className="mx-auto max-w-[1600px] scroll-mt-20 px-4 py-16 sm:px-6">
          <SectionTitle eyebrow="The dataset" title="Voice library">
            All {recordings.length || ""} clips from the ASVspoof 2019 LA subset. Their real/fake answers are kept
            hidden on purpose — judge first, then check the protocol file.
          </SectionTitle>
          <VoiceLibrary
            recordings={recordings}
            scores={scores}
            threshold={liveProjection?.threshold ?? null}
            selectedId={selectedId}
            onSelect={(recordingId) => select(recordingId, true)}
          />
        </section>

        <footer className="mx-auto max-w-[1600px] border-t border-white/10 px-4 py-8 text-xs text-slate-500 sm:px-6">
          Images from Wikimedia Commons: VODER photographs from Bell
          Telephone Quarterly, 1940 (no known restrictions); Praat waveform by Wugapodes (LGPL); speech spectrogram by
          Kelly Fitz (CC BY-SA 3.0); Pleiades by NASA, ESA, AURA/Caltech (public domain). Full links in
          public/deepfake/CREDITS.md.
        </footer>
      </main>
    </div>
  );
};

const Hero = ({
  onStart,
  onOwnVoice,
  onSurprise,
  ready,
  modelLabel,
  blurb,
}: {
  onStart: () => void;
  onOwnVoice: () => void;
  onSurprise: () => void;
  ready: boolean;
  modelLabel: string;
  blurb?: string;
}) => (
  <section className="mx-auto grid max-w-[1600px] items-center gap-10 px-4 pb-6 pt-12 sm:px-6 lg:grid-cols-[1fr_1.1fr] lg:pt-20">
    <div>
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-4 inline-flex items-center gap-2 rounded-full bg-white/5 px-3 py-1 text-xs text-slate-300 ring-1 ring-white/10"
      >
        <Sparkles className="h-3.5 w-3.5 text-violet-300" /> Audio deepfake detection
      </motion.div>
      <motion.h1
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.2, 0.7, 0.3, 1] }}
        className="font-display text-5xl font-bold leading-[1.05] text-white sm:text-6xl"
      >
        Real voice,
        <br />
        <span className="df-gradient-text">or a machine?</span>
      </motion.h1>
      <motion.p
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="mt-5 max-w-xl text-lg text-slate-300"
      >
        Synthetic speech is getting hard to hear. Listen to a clip, make your guess, then see what an AI detector
        thinks — and why.
      </motion.p>

      <ol className="mt-6 grid gap-3 sm:grid-cols-3">
        {[
          { icon: Ear, title: "Listen", text: "Pick a clip and play it" },
          { icon: UserRound, title: "Guess", text: "Real person or machine?" },
          { icon: Bot, title: "Compare", text: `Ask ${modelLabel.replace(/\s*\(Model [A-Z]\)/, "")}` },
        ].map((step, index) => (
          <motion.li
            key={step.title}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.25 + index * 0.1 }}
            className="df-glass rounded-2xl p-3"
          >
            <step.icon className="h-5 w-5 text-cyan-300" />
            <div className="mt-2 text-sm font-semibold text-white">
              {index + 1}. {step.title}
            </div>
            <div className="text-xs text-slate-400">{step.text}</div>
          </motion.li>
        ))}
      </ol>

      <div className="mt-7 flex flex-wrap items-center gap-3">
        <PrimaryButton onClick={onStart}>
          Start listening <ArrowDown className="h-4 w-4" />
        </PrimaryButton>
        <button
          type="button"
          onClick={onSurprise}
          disabled={!ready}
          className="inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold text-white ring-1 ring-white/20 transition hover:bg-white/10 disabled:opacity-40"
        >
          <Dices className="h-4 w-4" /> Random clip
        </button>
        <button
          type="button"
          onClick={onOwnVoice}
          className="inline-flex items-center gap-2 rounded-full px-5 py-2.5 text-sm font-semibold text-white ring-1 ring-white/20 transition hover:bg-white/10"
        >
          <Mic className="h-4 w-4" /> Try your own voice
        </button>
      </div>
      {blurb && <p className="mt-3 text-xs text-slate-500">Current detector: {modelLabel} — {blurb}.</p>}
    </div>

    {/* real vs machine */}
    <div className="relative grid grid-cols-2 gap-4">
      <motion.div initial={{ opacity: 0, x: -40, rotate: -4 }} animate={{ opacity: 1, x: 0, rotate: -2 }} transition={{ type: "spring", stiffness: 60, damping: 14 }}>
        <motion.div animate={{ y: [0, -8, 0] }} transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}>
          <figure className="df-well h-72 overflow-hidden rounded-2xl border border-white/10 ring-2 ring-cyan-300/40 sm:h-96">
            <HumanVoiceArt />
          </figure>
          <div className="mt-3 flex items-center gap-2 text-sm font-semibold text-cyan-300">
            <UserRound className="h-4 w-4" /> A real person, recorded
          </div>
        </motion.div>
      </motion.div>
      <motion.div className="mt-10" initial={{ opacity: 0, x: 40, rotate: 4 }} animate={{ opacity: 1, x: 0, rotate: 2 }} transition={{ type: "spring", stiffness: 60, damping: 14, delay: 0.1 }}>
        <motion.div animate={{ y: [0, 8, 0] }} transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}>
          <figure className="df-well h-72 overflow-hidden rounded-2xl border border-white/10 ring-2 ring-rose-300/40 sm:h-96">
            <MachineVoiceArt />
          </figure>
          <div className="mt-3 flex items-center gap-2 text-sm font-semibold text-rose-300">
            <Bot className="h-4 w-4" /> A voice made by a machine
          </div>
        </motion.div>
      </motion.div>
      <div className="pointer-events-none absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
        <motion.div
          className="grid h-16 w-16 df-canvas place-items-center rounded-full font-display text-lg font-bold text-white ring-1 ring-white/20"
          initial={{ scale: 0 }}
          animate={{ scale: 1 }}
          transition={{ type: "spring", stiffness: 200, damping: 12, delay: 0.5 }}
          style={{ boxShadow: "0 8px 24px -8px rgba(0,0,0,0.45)" }}
        >
          VS
        </motion.div>
      </div>
    </div>
  </section>
);

export default DeepfakePage;

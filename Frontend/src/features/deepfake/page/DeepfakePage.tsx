import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useScroll, useSpring } from "motion/react";
import { ArrowDown, AudioWaveform, Bot, Dices, Ear, HelpCircle, Lock, Mic, UserRound } from "lucide-react";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { API_BASE } from "@/lib/api";
import type { TaskDefinition } from "@/tasks/types";
import type { DeepfakeEmbeddingProjection, EmbeddingRecording, RecordingInfo, UserClip } from "../types";
import { AllDetectors } from "./AllDetectors";
import { errorMessage, listUserClips, postDeepfake } from "./api";
import { DetectorReport } from "./DetectorReport";
import { ListeningHeatmap } from "./ListeningHeatmap";
import { readSession, useSessionState, writeSession } from "./session";
import { SilenceTest } from "./SilenceTest";
import { ErrorNote, FeatureImage, PrimaryButton, SectionTitle } from "./ui";
import { VerdictPanel } from "./VerdictPanel";
import { VoiceLibrary } from "./VoiceLibrary";
import { VoiceMap, type ReductionMethod } from "./VoiceMap";
import { YourClips } from "./YourClips";
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
export const DeepfakePage = ({ task }: { task: TaskDefinition }) => {
  const availableModels = task.models.filter((option) => option.available);
  const fallbackModel = task.defaultModel ?? availableModels[0]?.id ?? "";
  // Everything the visitor chose is restored after a refresh (see session.ts).
  const [storedModel, setModel] = useSessionState("model", fallbackModel);
  // A remembered model that is no longer offered falls back to the default.
  const model = availableModels.some((option) => option.id === storedModel) ? storedModel : fallbackModel;
  const modelLabel = task.models.find((option) => option.id === model)?.label ?? model;

  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useSessionState("selected", "");
  const [userClips, setUserClips] = useState<UserClip[]>([]);

  const [mapStarted, setMapStarted] = useSessionState("map.started", false);
  const [method, setMethod] = useSessionState<ReductionMethod>("map.method", "pca");
  const [is3D, setIs3D] = useSessionState("map.3d", false);
  const [refreshToken, setRefreshToken] = useState(0);
  const [projection, setProjection] = useState<DeepfakeEmbeddingProjection | null>(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);

  const stage = useRef<HTMLElement>(null);

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

  const projectionKey = `map.projection.${model}.${method}.${is3D ? 3 : 2}.${userClipKey}`;
  const lastRefresh = useRef(refreshToken);

  useEffect(() => {
    if (!mapStarted || recordings.length === 0) return;
    // A map already drawn for exactly these settings comes back from the
    // session; the refresh button is the way to ask the server again.
    const refreshing = lastRefresh.current !== refreshToken;
    lastRefresh.current = refreshToken;
    const cached = refreshing ? null : readSession<DeepfakeEmbeddingProjection>(projectionKey);
    if (cached) {
      setProjection(cached);
      setMapError(null);
      setMapLoading(false);
      return;
    }
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
        writeSession(projectionKey, payload);
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

  // Come back to the same place on the page after a refresh. The browser's
  // own restoration fires before the async content has height, so the
  // position is saved on the way out and re-applied once the clips load.
  useEffect(() => {
    const save = () => writeSession("scroll", window.scrollY);
    window.addEventListener("pagehide", save);
    return () => window.removeEventListener("pagehide", save);
  }, []);
  const scrollRestored = useRef(false);
  useEffect(() => {
    if (scrollRestored.current || recordings.length === 0) return;
    scrollRestored.current = true;
    const target = readSession<number>("scroll");
    if (!target) return;
    const timer = window.setTimeout(() => window.scrollTo({ top: target }), 150);
    return () => window.clearTimeout(timer);
  }, [recordings.length]);

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
    <div className="df-page relative min-h-screen overflow-x-hidden">
      <motion.div
        className="fixed inset-x-0 top-0 z-50 h-0.5 origin-left df-accent-bar"
        style={{ scaleX: progress }}
        aria-hidden
      />

      {/* Toolbar — the same furniture as the other task pages. */}
      <header className="sticky top-0 z-40 border-b border-border bg-card">
        <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2">
          <Link to="/" className="flex items-center gap-1.5 text-base font-bold text-white">
            <AudioWaveform className="h-4 w-4 text-cyan-300" />
            VoxLIT
          </Link>
          <span className="rounded-sm bg-white/5 px-2 py-0.5 text-xs font-medium text-slate-300 ring-1 ring-white/10">
            {task.name}
          </span>

          <div className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400">Model:</span>
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="cursor-help">
                    <HelpCircle className="h-3.5 w-3.5 text-slate-400" />
                  </span>
                </TooltipTrigger>
                <TooltipContent className="max-w-xs text-xs font-normal">
                  {MODEL_BLURBS[model] ? `${modelLabel}: ${MODEL_BLURBS[model]}.` : modelLabel}
                </TooltipContent>
              </Tooltip>
            </TooltipProvider>
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger className="h-7 w-[220px] text-xs" aria-label="Detector model">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                {availableModels.map((option) => (
                  <SelectItem key={option.id} value={option.id} className="text-xs">
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-1.5">
            <span className="text-xs text-slate-400">Dataset:</span>
            <span className="rounded-sm border border-border px-2 py-1 text-xs text-slate-300">
              {task.datasets.find((dataset) => dataset.available)?.label ?? "None"}
            </span>
          </div>

          <nav className="ml-auto hidden items-center gap-0.5 lg:flex" aria-label="Sections">
            {NAV.map((item) => (
              <a
                key={item.href}
                href={item.href}
                className="rounded-sm px-2 py-1 text-xs text-slate-400 transition hover:bg-white/5 hover:text-white"
              >
                {item.label}
              </a>
            ))}
            <Link
              to="/help"
              className="rounded-sm px-2 py-1 text-xs text-cyan-300 font-medium transition hover:bg-white/10 hover:text-white flex items-center gap-1"
            >
              <HelpCircle className="h-3 w-3" />
              Help & Formulas
            </Link>
          </nav>
        </div>
      </header>

      <main className="relative z-10">
        <Hero
          onStart={() => stage.current?.scrollIntoView({ behavior: "smooth", block: "start" })}
          onOwnVoice={() => document.getElementById("your-voice")?.scrollIntoView({ behavior: "smooth", block: "start" })}
          onSurprise={surprise}
          ready={recordings.length > 0}
          modelLabel={modelLabel}
        />

        {listError && (
          <div className="mx-auto max-w-[1600px] px-4">
            <ErrorNote>{listError}</ErrorNote>
          </div>
        )}

        <section id="your-voice" className="mx-auto max-w-[1600px] scroll-mt-14 px-4 py-4">
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
        <section id="listen" ref={stage} className="mx-auto max-w-[1600px] scroll-mt-14 px-4 py-4">
          <SectionTitle eyebrow="Step 1: Listen" title={<>Pick a voice. <span className="df-gradient-text">Trust your ears?</span></>}>
            Hover a point to peek at a clip, click one to study it. Its verdict appears on the left and the
            explanations on the right.
          </SectionTitle>

          <div className="grid gap-3 xl:grid-cols-[minmax(300px,360px)_minmax(0,1fr)_minmax(300px,380px)]">
            <motion.aside
              initial={{ opacity: 0, x: -30 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true }}
              className={`df-glass ${selected ? "df-glow-border" : ""}`}
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
              className="df-glass order-first xl:order-none"
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
              className="df-glass"
              aria-label="Explain this clip"
            >
              <div className="df-panel-head px-3 py-2">
                <div className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">Step 2: Look closer</div>
                <h3 className="text-sm font-bold text-white">Why does it think so?</h3>
              </div>
              <div className="p-3">
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
                      { src: "/deepfake/saliency.png", title: "Where did it listen?", text: "A heatmap of the moments that pushed it towards synthetic." },
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
              </div>
            </motion.aside>
          </div>
        </section>

        <section id="compare" className="mx-auto max-w-[1600px] scroll-mt-14 px-4 py-4">
          <SectionTitle eyebrow="Step 3: Second opinions" title={<>Ask <span className="df-gradient-text">all {availableModels.length} detectors</span></>}>
            Every detector hears the same clip differently. Run the verdict, the silence test and the heatmap on each of
            them and compare.
          </SectionTitle>
          {selected ? (
            <AllDetectors models={availableModels} recording={selected} />
          ) : (
            <p className="df-glass p-4 text-center text-sm text-slate-400">
              Pick a clip above, or upload or record your own, to compare the detectors on it.
            </p>
          )}
        </section>

        <section id="report" className="mx-auto max-w-[1600px] scroll-mt-14 px-4 py-4">
          <SectionTitle eyebrow="Step 4: The big picture" title={<>How good is <span className="df-gradient-text">{modelLabel}</span>?</>}>
            A single verdict can be lucky. Test the detector on every labelled clip and see the mistakes it makes.
          </SectionTitle>
          <DetectorReport model={model} modelLabel={modelLabel} datasetSize={recordings.length} />
        </section>

        <section id="library" className="mx-auto max-w-[1600px] scroll-mt-14 px-4 py-4">
          <SectionTitle eyebrow="The dataset" title="Voice library">
            All {recordings.length || ""} clips from the ASVspoof 2019 LA subset. Their real/fake answers are kept
            hidden on purpose. Judge first, then check the protocol file.
          </SectionTitle>
          <VoiceLibrary
            recordings={recordings}
            scores={scores}
            threshold={liveProjection?.threshold ?? null}
            selectedId={selectedId}
            onSelect={(recordingId) => select(recordingId, true)}
          />
        </section>

        <footer className="mx-auto max-w-[1600px] border-t border-border px-4 py-4 text-xs text-slate-500">
          Feature images from Wikimedia Commons: VODER photograph from Bell Telephone Quarterly, 1940 (no known
          restrictions); Praat waveform by Wugapodes (LGPL); speech spectrogram by Kelly Fitz (CC BY-SA 3.0). Full
          links in public/deepfake/CREDITS.md.
        </footer>
      </main>
    </div>
  );
};

/**
 * The task intro, set out like the opening of a paper: title and abstract
 * with the actions beneath, and the three-step protocol beside it as a short
 * method timeline. Model and dataset already sit in the header bar, so they
 * are not repeated here.
 */
const Hero = ({
  onStart,
  onOwnVoice,
  onSurprise,
  ready,
  modelLabel,
}: {
  onStart: () => void;
  onOwnVoice: () => void;
  onSurprise: () => void;
  ready: boolean;
  modelLabel: string;
}) => {
  const detector = modelLabel.replace(/\s*\(Model [A-Z]\)/, "");
  const steps = [
    { icon: Ear, title: "Listen", text: "Select and play an utterance from the evaluation set." },
    { icon: UserRound, title: "Guess", text: "Make a blind judgement: bona fide or spoofed?" },
    { icon: Bot, title: "Compare", text: `Check it against ${detector}'s score, threshold and attribution.` },
  ];

  return (
    <section className="mx-auto max-w-[1600px] px-4 pb-2 pt-5">
      <div className="df-glass grid gap-6 p-5 lg:grid-cols-[minmax(0,1fr)_minmax(280px,380px)] lg:gap-10 lg:p-6">
        <div className="min-w-0">
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">
            Audio anti-spoofing: interactive analysis
          </div>
          <h1 className="mt-1 font-display text-2xl font-bold tracking-tight text-white">Audio Deepfake Detection</h1>
          <p className="mt-0.5 text-sm font-medium text-slate-400">
            Real voice, <span className="df-gradient-text">or a machine?</span>
          </p>
          <p className="mt-3 max-w-3xl text-sm leading-relaxed text-slate-300">
            Compare human judgement with automatic spoofing detectors on bona fide and synthetic speech. Each
            utterance comes with the detector&apos;s spoof score, its decision threshold, a time-frequency
            attribution map and its place in the model&apos;s embedding space. Ground-truth labels stay hidden, so
            every judgement is made blind.
          </p>

          <div className="mt-5 flex flex-wrap items-center gap-2">
            <PrimaryButton onClick={onStart}>
              Start listening <ArrowDown className="h-3.5 w-3.5" />
            </PrimaryButton>
            <button
              type="button"
              onClick={onSurprise}
              disabled={!ready}
              className="inline-flex items-center gap-1.5 rounded-sm border border-border px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-white/5 disabled:opacity-40"
            >
              <Dices className="h-3.5 w-3.5" /> Random utterance
            </button>
            <button
              type="button"
              onClick={onOwnVoice}
              className="inline-flex items-center gap-1.5 rounded-sm border border-border px-3 py-1.5 text-xs font-semibold text-white transition hover:bg-white/5"
            >
              <Mic className="h-3.5 w-3.5" /> Test your own recording
            </button>
          </div>
        </div>

        {/* the protocol as a method timeline: numbered nodes joined by a rule */}
        <div className="lg:border-l lg:border-border lg:pl-8">
          <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-500">Protocol</div>
          <ol className="mt-3">
            {steps.map((step, index) => (
              <motion.li
                key={step.title}
                initial={{ opacity: 0, x: 8 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.05 + index * 0.08 }}
                className="relative flex gap-3 pb-4 last:pb-0"
              >
                {index < steps.length - 1 && (
                  <span aria-hidden className="absolute left-3 top-7 h-[calc(100%-1.75rem)] w-px bg-slate-300" />
                )}
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full border border-border bg-[hsl(var(--panel-background))] font-mono text-[11px] font-semibold text-slate-300">
                  {index + 1}
                </span>
                <div className="min-w-0 pt-0.5">
                  <div className="flex items-center gap-1.5 text-xs font-semibold text-white">
                    <step.icon className="h-3.5 w-3.5 text-cyan-300" aria-hidden />
                    <span>
                      {index + 1}. {step.title}
                    </span>
                  </div>
                  <div className="mt-0.5 text-xs leading-snug text-slate-400">{step.text}</div>
                </div>
              </motion.li>
            ))}
          </ol>
        </div>
      </div>
    </section>
  );
};

export default DeepfakePage;

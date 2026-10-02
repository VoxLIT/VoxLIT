import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { AnimatePresence, motion, useScroll, useSpring } from "motion/react";
import { ArrowDown, Bot, Dices, Ear, HelpCircle, Lock, Mic, Upload, UserRound } from "lucide-react";
import { toast } from "sonner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { API_BASE } from "@/lib/api";
import type { TaskDefinition } from "@/tasks/types";
import type { CustomDataset, DeepfakeEmbeddingProjection, EmbeddingRecording, RecordingInfo, UserClip } from "../types";
import { AllDetectors } from "./AllDetectors";
import { errorMessage, listDatasetRecordings, listDatasets, listUserClips, postDeepfake, uploadUserClip } from "./api";
import { DatasetManager } from "./DatasetManager";
import { HelpFormulas } from "./HelpFormulas";
import { DetectorReport } from "./DetectorReport";
import { ListeningHeatmap } from "./ListeningHeatmap";
import { readSession, useSessionState, writeSession } from "./session";
import { SilenceTest } from "./SilenceTest";
import { ErrorNote, FeatureImage, PrimaryButton, SectionTitle } from "./ui";
import { VerdictPanel } from "./VerdictPanel";
import { VoiceLibrary } from "./VoiceLibrary";
import { VoiceMap, type ReductionMethod } from "./VoiceMap";
import { convertToWav, wavName } from "./wav";
import { YourClips } from "./YourClips";
import "./deepfake-page.css";

const MODEL_BLURBS: Record<string, string> = {
  "xlsr-deepfake": "listens to raw sound waves",
  "ast-fakeaudio": "reads the spectrogram like an image",
  "xlsr-mamba": "a state-space model over wav2vec2 features",
};

/** Dataset menu value for the built-in subset (custom ones use their name). */
const BUILTIN = "__builtin__";
const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

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

  const builtinDataset = task.datasets.find((dataset) => dataset.available);
  // "" = the built-in ASVspoof subset; otherwise a custom dataset's name.
  const [customDataset, setCustomDataset] = useSessionState("dataset", "");
  const [customDatasets, setCustomDatasets] = useState<CustomDataset[]>([]);
  const [datasetsVersion, setDatasetsVersion] = useState(0);
  const activeCustom = customDatasets.find((dataset) => dataset.dataset_name === customDataset) ?? null;
  const datasetLabel = customDataset || builtinDataset?.label || "None";

  const [recordings, setRecordings] = useState<RecordingInfo[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useSessionState("selected", "");
  const [userClips, setUserClips] = useState<UserClip[]>([]);
  const uploadInput = useRef<HTMLInputElement>(null);

  const [mapStarted, setMapStarted] = useSessionState("map.started", false);
  const [method, setMethod] = useSessionState<ReductionMethod>("map.method", "pca");
  const [is3D, setIs3D] = useSessionState("map.3d", false);
  const [refreshToken, setRefreshToken] = useState(0);
  const [projection, setProjection] = useState<DeepfakeEmbeddingProjection | null>(null);
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);

  const stage = useRef<HTMLElement>(null);

  // The visitor's own clips from earlier in this session. Optional: the page
  // works without them, so a failure here stays silent.
  useEffect(() => {
    listUserClips()
      .then(setUserClips)
      .catch(() => undefined);
  }, []);

  // Custom datasets for the toolbar's Dataset menu; a remembered one that has
  // since expired or been deleted falls back to the built-in subset.
  useEffect(() => {
    listDatasets()
      .then(({ datasets }) => {
        setCustomDatasets(datasets);
        if (customDataset && !datasets.some((dataset) => dataset.dataset_name === customDataset)) setCustomDataset("");
      })
      .catch(() => undefined);
    // customDataset is read once per refresh on purpose.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [datasetsVersion]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        let list: RecordingInfo[];
        if (customDataset) {
          list = await listDatasetRecordings(customDataset);
        } else {
          const response = await fetch(`${API_BASE}/tasks/deepfake/dataset/recordings`, { credentials: "include" });
          const payload = await response.json();
          if (!response.ok) throw new Error(payload.detail || "Could not list recordings.");
          list = payload.recordings as RecordingInfo[];
        }
        if (cancelled) return;
        setRecordings(list);
        setListError(null);
      } catch (caught) {
        if (cancelled) return;
        setRecordings([]);
        setListError(errorMessage(caught, "Could not list recordings."));
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, [customDataset, datasetsVersion]);

  // Placed on the same map as the dataset, so the visitor can see which
  // dataset clips the detector thinks their voice resembles.
  const userClipIds = useMemo(() => userClips.map((clip) => clip.recording_id), [userClips]);
  const userClipKey = userClipIds.join(",");

  // The dataset's file count is part of the key, so adding files redraws the map.
  const datasetKey = customDataset ? `custom-${customDataset}-${recordings.length}` : "builtin";
  const projectionKey = `map.projection.${datasetKey}.${model}.${method}.${is3D ? 3 : 2}.${userClipKey}`;
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
        ...(customDataset ? { dataset: customDataset } : {}),
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
  }, [mapStarted, recordings.length, model, method, is3D, refreshToken, userClipKey, datasetKey]);

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

  const changeDataset = (value: string) => {
    setCustomDataset(value === BUILTIN ? "" : value);
    setSelectedId("");
    setProjection(null);
  };

  // The toolbar's Upload: same path as "Test your own voice" (converted to
  // 16 kHz WAV in the browser, then stored as one of the visitor's clips).
  const uploadFromToolbar = async (files: FileList | null) => {
    for (const file of Array.from(files ?? [])) {
      // Same limits as the "Test your own voice" uploader.
      if (file.size > MAX_UPLOAD_BYTES) {
        toast.error(`${file.name} is larger than 25 MB.`);
        continue;
      }
      if (!file.type.startsWith("audio/") && !/\.(wav|mp3|m4a|flac|ogg)$/i.test(file.name)) {
        toast.error(`Invalid file type: ${file.name}. Supported formats: WAV, MP3, M4A, FLAC, OGG`);
        continue;
      }
      try {
        const { wav } = await convertToWav(file);
        const clip = await uploadUserClip(wav, wavName(file.name), "upload");
        toast.success(`Uploaded: ${file.name}`);
        addUserClip(clip);
      } catch (caught) {
        toast.error(`Failed to upload ${file.name}: ${errorMessage(caught, "Unknown error")}`);
      }
    }
    if (uploadInput.current) uploadInput.current.value = "";
  };

  const { scrollYProgress } = useScroll();
  const progress = useSpring(scrollYProgress, { stiffness: 120, damping: 30 });

  return (
    <div className="df-page relative min-h-screen overflow-x-hidden">
      <motion.div
        className="fixed inset-x-0 top-0 z-50 h-0.5 origin-left df-accent-bar"
        style={{ scaleX: progress }}
        aria-hidden
      />

      {/* Toolbar: the same furniture, sizes and order as the other task pages'
          Toolbar (brand, task, Model, Dataset | Manage Datasets, Help, Upload),
          plus this page's section links. */}
      <TooltipProvider>
        <header className="df-toolbar sticky top-0 z-40 border-b border-border bg-white">
          <div className="flex min-h-12 flex-wrap items-center justify-between gap-x-5 gap-y-2 px-5 py-2">
            <div className="flex flex-wrap items-center gap-5">
              <div className="flex items-center gap-2.5">
                <Link to="/" className="text-base font-bold text-foreground transition-colors hover:text-primary">
                  VoxLIT
                </Link>
                <Badge variant="outline" className="border-primary/20 bg-primary/10 text-[10px] text-primary">
                  v1.0
                </Badge>
                <Badge variant="secondary" className="text-[10px]">
                  {task.name}
                </Badge>
              </div>

              <div className="flex flex-wrap items-center gap-3">
                <div className="flex items-center gap-1.5">
                  <div className="flex items-center gap-1">
                    <span className="text-xs font-medium text-foreground">Model:</span>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle className="h-3 w-3 cursor-help text-muted-foreground transition-colors hover:text-primary" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs space-y-1">
                        <p className="text-xs">{MODEL_BLURBS[model] ? `${modelLabel}: ${MODEL_BLURBS[model]}.` : modelLabel}</p>
                        <p className="text-xs">Every view on the page follows this detector.</p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Select value={model} onValueChange={setModel}>
                    <SelectTrigger className="h-7 w-[220px] border-border text-xs" aria-label="Detector model">
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
                  <div className="flex items-center gap-1">
                    <span className="text-xs font-medium text-foreground">Dataset:</span>
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <HelpCircle className="h-3 w-3 cursor-help text-muted-foreground transition-colors hover:text-primary" />
                      </TooltipTrigger>
                      <TooltipContent className="max-w-xs space-y-1">
                        <p className="text-xs">Select the audio dataset to analyse.</p>
                        <p className="text-xs">
                          The built-in subset is labelled; custom datasets are yours, and need a label file before the
                          Detector report can measure EER on them.
                        </p>
                      </TooltipContent>
                    </Tooltip>
                  </div>
                  <Select value={customDataset || BUILTIN} onValueChange={changeDataset}>
                    <SelectTrigger className="h-7 w-48 border-border text-xs" aria-label="Dataset">
                      <SelectValue placeholder="To be added" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value={BUILTIN} className="text-xs" disabled={!builtinDataset}>
                        {builtinDataset?.label ?? "Built-in dataset"}
                      </SelectItem>
                      {customDatasets.length > 0 && (
                        <>
                          <SelectItem disabled value="separator" className="text-xs">
                            ── Custom Datasets ──
                          </SelectItem>
                          {customDatasets.map((dataset) => (
                            <SelectItem
                              key={dataset.dataset_name}
                              value={dataset.dataset_name}
                              disabled={dataset.total_files === 0}
                              className="text-xs"
                            >
                              {dataset.dataset_name} ({dataset.total_files})
                            </SelectItem>
                          ))}
                        </>
                      )}
                    </SelectContent>
                  </Select>
                </div>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <nav className="hidden items-center gap-0.5 2xl:flex" aria-label="Sections">
                {NAV.map((item) => (
                  <a
                    key={item.href}
                    href={item.href}
                    className="rounded-sm px-2 py-1 text-xs text-muted-foreground transition hover:bg-muted hover:text-foreground"
                  >
                    {item.label}
                  </a>
                ))}
              </nav>

              <DatasetManager
                activeDataset={customDataset || null}
                onSelect={(name) => {
                  changeDataset(name);
                  setDatasetsVersion((version) => version + 1);
                }}
                onChanged={(event) => {
                  if (event.type === "deleted" && event.datasetName === customDataset) changeDataset(BUILTIN);
                  setDatasetsVersion((version) => version + 1);
                }}
              />

              <HelpFormulas />

              <Tooltip>
                <TooltipTrigger asChild>
                  <Button variant="default" size="sm" className="h-7 text-xs shadow-aws-sm" onClick={() => uploadInput.current?.click()}>
                    <Upload className="mr-1.5 h-3.5 w-3.5" />
                    Upload
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Upload audio files to test (added to "Your voice")</p>
                </TooltipContent>
              </Tooltip>
              <input
                ref={uploadInput}
                type="file"
                accept="audio/*,.flac,.wav,.mp3,.m4a,.ogg"
                multiple
                onChange={(event) => uploadFromToolbar(event.target.files)}
                className="hidden"
              />
            </div>
          </div>
        </header>
      </TooltipProvider>

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
          <DetectorReport
            key={customDataset || BUILTIN}
            model={model}
            modelLabel={modelLabel}
            datasetSize={recordings.length}
            customDataset={customDataset || null}
            labelledCount={activeCustom?.labels.matched_files ?? null}
          />
        </section>

        <section id="library" className="mx-auto max-w-[1600px] scroll-mt-14 px-4 py-4">
          <SectionTitle eyebrow="The dataset" title={customDataset ? <>Voice library: <span className="df-gradient-text">{customDataset}</span></> : "Voice library"}>
            {customDataset
              ? `All ${recordings.length} clips in your dataset "${customDataset}". Any labels you uploaded stay hidden here and are used only for the Detector report's aggregate metrics.`
              : `All ${recordings.length || ""} clips from the ${datasetLabel}. Their real/fake answers are kept hidden on purpose. Judge first, then check the protocol file.`}
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

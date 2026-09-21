import { useMemo, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Pause, Play, Search } from "lucide-react";
import type { EmbeddingRecording, RecordingInfo } from "../types";
import { audioUrlFor, formatBytes, formatSeconds } from "./api";
import { preview, usePreviewUrl } from "./audio";
import { usePalette } from "./theme";

type Sort = "name" | "length" | "score";
type Lean = "all" | "real" | "synthetic";

interface VoiceLibraryProps {
  recordings: RecordingInfo[];
  /** The map's per-clip scores, once the map has been drawn for this model. */
  scores: Map<string, EmbeddingRecording>;
  threshold: number | null;
  selectedId: string;
  onSelect: (recordingId: string) => void;
}

/**
 * The whole dataset laid out on the page — no inner scroll box. Every clip is
 * a card you can preview in place or send up to the verdict panel.
 */
export const VoiceLibrary = ({ recordings, scores, threshold, selectedId, onSelect }: VoiceLibraryProps) => {
  const { scoreColor } = usePalette();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<Sort>("name");
  const [lean, setLean] = useState<Lean>("all");
  const playingUrl = usePreviewUrl();
  const scored = scores.size > 0 && threshold !== null;
  const longest = useMemo(() => Math.max(1, ...recordings.map((recording) => recording.duration_seconds ?? 0)), [recordings]);

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const rows = recordings.filter((recording) => {
      if (needle && !recording.display_filename.toLowerCase().includes(needle)) return false;
      if (lean === "all" || !scored) return true;
      const score = scores.get(recording.recording_id)?.spoof_probability;
      if (score === undefined) return false;
      return lean === "synthetic" ? score >= threshold! : score < threshold!;
    });
    const by: Record<Sort, (a: RecordingInfo, b: RecordingInfo) => number> = {
      name: (a, b) => a.display_filename.localeCompare(b.display_filename),
      length: (a, b) => (b.duration_seconds ?? 0) - (a.duration_seconds ?? 0),
      score: (a, b) =>
        (scores.get(b.recording_id)?.spoof_probability ?? -1) - (scores.get(a.recording_id)?.spoof_probability ?? -1),
    };
    return [...rows].sort(by[sort]);
  }, [recordings, query, sort, lean, scores, scored, threshold]);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <label className="relative min-w-[220px] flex-1">
          <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-400" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search clips by name…"
            aria-label="Search clips"
            className="w-full rounded-full border border-white/10 bg-white/5 py-2.5 pl-9 pr-4 text-sm text-white placeholder:text-slate-500 focus:border-cyan-300/60 focus:outline-none"
          />
        </label>
        <Segmented
          label="Sort"
          value={sort}
          onChange={(next) => setSort(next as Sort)}
          options={[
            { id: "name", label: "Name" },
            { id: "length", label: "Longest" },
            ...(scored ? [{ id: "score" as Sort, label: "Most synthetic" }] : []),
          ]}
        />
        {scored && (
          <Segmented
            label="Filter"
            value={lean}
            onChange={(next) => setLean(next as Lean)}
            options={[
              { id: "all", label: "All" },
              { id: "real", label: "Sounds real" },
              { id: "synthetic", label: "Sounds synthetic" },
            ]}
          />
        )}
        <span className="text-sm text-slate-400">
          {visible.length} of {recordings.length} clips
        </span>
      </div>
      {!scored && (
        <p className="text-xs text-slate-500">Map the voices above to see each clip&apos;s score here and filter by it.</p>
      )}

      <motion.ul layout className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5">
        <AnimatePresence initial={false}>
          {visible.map((recording, index) => {
            const point = scores.get(recording.recording_id);
            // Unscored clips share the neutral accent; the score only colours a card once the map has run.
            const colour = point ? scoreColor(point.spoof_probability) : "var(--df-accent)";
            const synthetic = point && threshold !== null ? point.spoof_probability >= threshold : null;
            const lengthShare = Math.min(1, (recording.duration_seconds ?? 0) / longest);
            const selected = recording.recording_id === selectedId;
            const url = audioUrlFor(recording.recording_id);
            const playing = playingUrl === url;
            return (
              <motion.li
                layout
                key={recording.recording_id}
                initial={{ opacity: 0, y: 16, scale: 0.96 }}
                whileInView={{ opacity: 1, y: 0, scale: 1 }}
                viewport={{ once: true, margin: "-40px" }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.35, delay: (index % 10) * 0.025 }}
              >
                <div
                  className={`group relative flex items-center gap-3 overflow-hidden rounded-2xl p-3 transition-all ${
                    selected ? "bg-white/10 ring-2" : "bg-white/[0.035] ring-1 ring-white/10 hover:bg-white/[0.07]"
                  }`}
                  style={selected ? { boxShadow: `0 0 30px -8px ${colour}`, ["--tw-ring-color" as string]: colour } : undefined}
                >
                  {/* score wash + stripe */}
                  <span
                    className="pointer-events-none absolute inset-0"
                    style={{ background: `linear-gradient(90deg, color-mix(in srgb, ${colour} ${point ? 14 : 7}%, transparent), transparent 70%)` }}
                    aria-hidden
                  />
                  <span className="absolute inset-y-0 left-0 w-1" style={{ background: colour }} aria-hidden />
                  {/* relative clip length */}
                  <span
                    className="pointer-events-none absolute bottom-0 left-1 h-0.5 opacity-60"
                    style={{ width: `calc((100% - 0.25rem) * ${lengthShare})`, background: colour }}
                    aria-hidden
                  />
                  <button
                    type="button"
                    onClick={() => preview.toggle(url)}
                    aria-label={`${playing ? "Stop" : "Preview"} ${recording.display_filename}`}
                    className="relative grid h-10 w-10 shrink-0 place-items-center rounded-full transition hover:scale-110"
                    style={{
                      color: colour,
                      background: `color-mix(in srgb, ${colour} 16%, transparent)`,
                      boxShadow: `inset 0 0 0 1px color-mix(in srgb, ${colour} 35%, transparent)`,
                    }}
                  >
                    {playing && <span className="df-ripple absolute inset-0 rounded-full border border-cyan-300" aria-hidden />}
                    {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
                  </button>
                  <button
                    type="button"
                    onClick={() => onSelect(recording.recording_id)}
                    aria-pressed={selected}
                    className="min-w-0 flex-1 text-left"
                  >
                    <div className="truncate font-mono text-sm text-white">{recording.display_filename}</div>
                    <div className="flex items-center gap-2 text-[11px] text-slate-400">
                      <span>{formatSeconds(recording.duration_seconds)}</span>
                      <span>·</span>
                      <span>{formatBytes(recording.size_bytes)}</span>
                      {point && (
                        <span
                          className="ml-auto rounded-full px-1.5 py-px font-mono font-semibold"
                          style={{ color: colour, background: `color-mix(in srgb, ${colour} 14%, transparent)` }}
                          title={synthetic === null ? undefined : synthetic ? "Sounds synthetic" : "Sounds real"}
                        >
                          {point.spoof_probability.toFixed(2)}
                        </span>
                      )}
                    </div>
                  </button>
                </div>
              </motion.li>
            );
          })}
        </AnimatePresence>
      </motion.ul>
    </div>
  );
};

function Segmented<T extends string>({
  label,
  value,
  onChange,
  options,
}: {
  label: string;
  value: T;
  onChange: (value: T) => void;
  options: { id: T; label: string }[];
}) {
  return (
    <div className="flex rounded-full bg-white/5 p-1 ring-1 ring-white/10" role="radiogroup" aria-label={label}>
      {options.map((option) => (
        <button
          key={option.id}
          type="button"
          role="radio"
          aria-checked={value === option.id}
          onClick={() => onChange(option.id)}
          className="relative rounded-full px-3 py-1 text-xs font-semibold text-slate-300 transition-colors hover:text-white aria-checked:text-slate-950"
        >
          {value === option.id && (
            <motion.span layoutId={`df-seg-${label}`} className="df-pill absolute inset-0 rounded-full" transition={{ type: "spring", stiffness: 400, damping: 30 }} />
          )}
          <span className="relative">{option.label}</span>
        </button>
      ))}
    </div>
  );
}

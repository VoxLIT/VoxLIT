import { useMemo, useState } from "react";
import { motion } from "motion/react";
import { ArrowDown, ArrowUp, ChevronsUpDown, Pause, Play, Search } from "lucide-react";
import type { EmbeddingRecording, RecordingInfo } from "../types";
import { audioUrlFor, formatBytes, formatSeconds } from "./api";
import { preview, usePreviewUrl } from "./audio";
import { usePalette } from "./theme";
import { formatScore } from "./palette";

type Column = "name" | "length" | "size" | "score";
type Direction = "asc" | "desc";
type Lean = "all" | "real" | "synthetic";

interface VoiceLibraryProps {
  recordings: RecordingInfo[];
  /** The map's per-clip scores, once the map has been drawn for this model. */
  scores: Map<string, EmbeddingRecording>;
  threshold: number | null;
  selectedId: string;
  onSelect: (recordingId: string) => void;
}

const COLUMNS: { id: Column; label: string; align: "left" | "right"; numeric: boolean }[] = [
  { id: "name", label: "Clip", align: "left", numeric: false },
  { id: "length", label: "Length", align: "right", numeric: true },
  { id: "size", label: "Size", align: "right", numeric: true },
  { id: "score", label: "Spoof score", align: "right", numeric: true },
];

/**
 * The dataset as a table — the whole of it, laid out on the page with no inner
 * scroll box, so scrolling the page reads the dataset. Sortable columns and a
 * search box for a researcher; a play button and plain "reads as" wording so
 * anyone can use it.
 *
 * The score column is the DETECTOR's output, and it is empty until the voice
 * map has been drawn; the dataset's own bona fide/spoof answer is never here.
 */
export const VoiceLibrary = ({ recordings, scores, threshold, selectedId, onSelect }: VoiceLibraryProps) => {
  const { scoreColor, ink } = usePalette();
  const [query, setQuery] = useState("");
  const [sort, setSort] = useState<{ column: Column; direction: Direction }>({ column: "name", direction: "asc" });
  const [lean, setLean] = useState<Lean>("all");
  const playingUrl = usePreviewUrl();
  const scored = scores.size > 0 && threshold !== null;

  const visible = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const rows = recordings.filter((recording) => {
      if (needle && !recording.display_filename.toLowerCase().includes(needle)) return false;
      if (lean === "all" || !scored) return true;
      const score = scores.get(recording.recording_id)?.spoof_probability;
      if (score === undefined) return false;
      return lean === "synthetic" ? score >= threshold! : score < threshold!;
    });
    const value: Record<Column, (recording: RecordingInfo) => number | string> = {
      name: (recording) => recording.display_filename,
      length: (recording) => recording.duration_seconds ?? 0,
      size: (recording) => recording.size_bytes,
      // unscored clips sort last in either direction
      score: (recording) => scores.get(recording.recording_id)?.spoof_probability ?? -1,
    };
    const read = value[sort.column];
    const factor = sort.direction === "asc" ? 1 : -1;
    return [...rows].sort((a, b) => {
      const left = read(a);
      const right = read(b);
      if (typeof left === "string" && typeof right === "string") return left.localeCompare(right) * factor;
      return ((left as number) - (right as number)) * factor;
    });
  }, [recordings, query, sort, lean, scores, scored, threshold]);

  const toggleSort = (column: Column) =>
    setSort((current) =>
      current.column === column
        ? { column, direction: current.direction === "asc" ? "desc" : "asc" }
        : { column, direction: column === "name" ? "asc" : "desc" },
    );

  return (
    <div className="df-glass">
      {/* filters */}
      <div className="df-panel-head flex flex-wrap items-center gap-2 px-3 py-2">
        <label className="relative min-w-[200px] flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-slate-400" />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search clips by name"
            aria-label="Search clips"
            className="h-7 w-full rounded-sm border border-border bg-white/5 pl-8 pr-2 text-xs text-white placeholder:text-slate-500 focus:border-cyan-300/60 focus:outline-none"
          />
        </label>

        {scored && (
          <div className="flex rounded-sm border border-border" role="radiogroup" aria-label="Filter by what the detector hears">
            {(
              [
                { id: "all", label: "All" },
                { id: "real", label: "Sounds real" },
                { id: "synthetic", label: "Sounds synthetic" },
              ] as const
            ).map((option) => (
              <button
                key={option.id}
                type="button"
                role="radio"
                aria-checked={lean === option.id}
                onClick={() => setLean(option.id)}
                className="px-2.5 py-1 text-xs font-medium text-slate-400 transition-colors first:rounded-l-sm last:rounded-r-sm hover:text-white aria-checked:bg-white/10 aria-checked:text-white"
              >
                {option.label}
              </button>
            ))}
          </div>
        )}

        <span className="text-xs tabular-nums text-slate-400">
          {visible.length} of {recordings.length} clips
        </span>
      </div>

      {!scored && (
        <p className="border-b border-border px-3 py-1.5 text-xs text-slate-500">
          The spoof score column fills in once the voice map has scored the dataset with this detector.
        </p>
      )}

      <motion.table
        initial={{ opacity: 0 }}
        whileInView={{ opacity: 1 }}
        viewport={{ once: true }}
        className="w-full border-collapse text-xs"
      >
        <thead>
          <tr className="border-b border-border text-left">
            <th scope="col" className="w-9 px-2 py-1.5">
              <span className="sr-only">Preview</span>
            </th>
            {COLUMNS.map((column) => {
              const active = sort.column === column.id;
              return (
                <th
                  key={column.id}
                  scope="col"
                  aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
                  className={`px-2 py-1.5 font-semibold ${column.align === "right" ? "text-right" : "text-left"}`}
                >
                  <button
                    type="button"
                    onClick={() => toggleSort(column.id)}
                    className={`inline-flex items-center gap-1 transition-colors hover:text-white ${
                      active ? "text-white" : "text-slate-400"
                    } ${column.align === "right" ? "flex-row-reverse" : ""}`}
                  >
                    {column.label}
                    {active ? (
                      sort.direction === "asc" ? (
                        <ArrowUp className="h-3 w-3" />
                      ) : (
                        <ArrowDown className="h-3 w-3" />
                      )
                    ) : (
                      <ChevronsUpDown className="h-3 w-3 opacity-50" />
                    )}
                  </button>
                </th>
              );
            })}
            <th scope="col" className="px-2 py-1.5 text-left font-semibold text-slate-400">
              Reads as
            </th>
          </tr>
        </thead>
        <tbody>
          {visible.map((recording) => {
            const point = scores.get(recording.recording_id);
            const synthetic = point && threshold !== null ? point.spoof_probability >= threshold : null;
            const colour = point ? scoreColor(point.spoof_probability) : undefined;
            const selected = recording.recording_id === selectedId;
            const url = audioUrlFor(recording.recording_id);
            const playing = playingUrl === url;
            return (
              <tr
                key={recording.recording_id}
                className={`border-b border-border/70 last:border-0 ${selected ? "bg-cyan-400/10" : "hover:bg-white/5"}`}
              >
                <td className="px-2 py-1">
                  <button
                    type="button"
                    onClick={() => preview.toggle(url)}
                    aria-label={`${playing ? "Stop" : "Preview"} ${recording.display_filename}`}
                    className="grid h-6 w-6 place-items-center rounded-sm text-slate-400 transition-colors hover:bg-white/10 hover:text-white"
                  >
                    {playing ? <Pause className="h-3.5 w-3.5" /> : <Play className="h-3.5 w-3.5" />}
                  </button>
                </td>
                <td className="px-2 py-1">
                  <button
                    type="button"
                    onClick={() => onSelect(recording.recording_id)}
                    aria-pressed={selected}
                    className={`max-w-full truncate font-mono transition-colors hover:text-cyan-300 ${
                      selected ? "font-semibold text-cyan-300" : "text-white"
                    }`}
                    title="Study this clip"
                  >
                    {recording.display_filename}
                  </button>
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-400">
                  {formatSeconds(recording.duration_seconds)}
                </td>
                <td className="px-2 py-1 text-right tabular-nums text-slate-400">{formatBytes(recording.size_bytes)}</td>
                <td className="px-2 py-1">
                  {point ? (
                    <div className="flex items-center justify-end gap-2">
                      {/* the score on a shared 0..1 axis, so the column reads as a distribution */}
                      <span className="hidden h-1.5 w-16 overflow-hidden rounded-sm sm:block" style={{ background: ink(0.08) }} aria-hidden>
                        <span
                          className="block h-full"
                          style={{ width: `${point.spoof_probability * 100}%`, background: colour }}
                        />
                      </span>
                      <span className="w-16 text-right font-mono tabular-nums" style={{ color: colour }}>
                        {formatScore(point.spoof_probability, 2)}
                      </span>
                    </div>
                  ) : (
                    <div className="text-right text-slate-500">n/a</div>
                  )}
                </td>
                <td className="px-2 py-1">
                  {synthetic === null ? (
                    <span className="text-slate-500" title="Score this dataset with the voice map above">
                      &mdash;
                    </span>
                  ) : (
                    <span className="inline-flex items-center gap-1.5" style={{ color: colour }}>
                      <span className="h-1.5 w-1.5 rounded-full" style={{ background: colour }} />
                      {synthetic ? "Sounds synthetic" : "Sounds real"}
                    </span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </motion.table>

      {visible.length === 0 && (
        <p className="px-3 py-6 text-center text-xs text-slate-500">No clip matches that search.</p>
      )}
    </div>
  );
};

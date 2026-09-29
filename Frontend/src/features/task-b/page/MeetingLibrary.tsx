import { DragEvent, useRef, useState, useSyncExternalStore } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Check, FolderOpen, Loader2, Pause, Play, Upload } from "lucide-react";
import type { RecordingInfo } from "../types";
import { audioUrlFor } from "./useDiarization";

// Mirrors the backend's limits (task_b/uploads.py) so a drag-drop, which
// skips the file picker's `accept` filter, fails here instead of after a
// 50 MB upload.
const ALLOWED_EXTENSIONS = ["wav", "mp3", "m4a", "flac"];
const MAX_FILE_BYTES = 50 * 1024 * 1024;
const UPLOAD_ID_PREFIX = "upl_";

const formatBytes = (bytes: number): string => {
  if (bytes >= 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${bytes} B`;
};

/* One shared element for card previews, so starting one stops the previous. */
let previewElement: HTMLAudioElement | null = null;
let previewUrl: string | null = null;
const listeners = new Set<() => void>();
const notify = () => listeners.forEach((listener) => listener());

const togglePreview = (url: string) => {
  if (typeof Audio === "undefined") return;
  if (!previewElement) {
    previewElement = new Audio();
    previewElement.addEventListener("ended", () => {
      previewUrl = null;
      notify();
    });
  }
  if (previewUrl === url && !previewElement.paused) {
    previewElement.pause();
    previewUrl = null;
  } else {
    previewElement.src = url;
    previewElement.currentTime = 0;
    try {
      (previewElement.play() as Promise<void> | undefined)?.catch(() => undefined);
    } catch {
      // playback refused (autoplay policy, no media support) — stay silent
    }
    previewUrl = url;
  }
  notify();
};

const usePreviewUrl = () =>
  useSyncExternalStore(
    (listener) => {
      listeners.add(listener);
      return () => listeners.delete(listener);
    },
    () => previewUrl,
    () => null,
  );

interface MeetingLibraryProps {
  recordings: RecordingInfo[];
  uploads: RecordingInfo[];
  selectedId: string;
  onSelect: (recordingId: string) => void;
  onUpload: (file: File) => void;
  isUploading: boolean;
  /** Selecting or uploading mid-run would orphan the run's result. */
  disabled: boolean;
  /** True until the demo dataset listing has come back (or failed). */
  isLoading: boolean;
}

/**
 * Step 1: two columns of one fixed height — meeting cards on the left,
 * scrolling inside that height, and the upload box on the right filling it.
 * Stacks on narrow screens. This is the page's only uploader.
 */
export const MeetingLibrary = ({
  recordings,
  uploads,
  selectedId,
  onSelect,
  onUpload,
  isUploading,
  disabled,
  isLoading,
}: MeetingLibraryProps) => {
  const playingUrl = usePreviewUrl();
  const all = [...recordings, ...uploads];

  return (
    <div className="grid gap-5 lg:h-[22rem] lg:grid-cols-[minmax(0,2fr)_minmax(260px,1fr)]">
      {/* p-1 leaves room for the selected card's 1.02 scale inside the scroll clip. */}
      <motion.ul
        layout
        className="grid max-h-[22rem] content-start gap-3 overflow-y-auto p-1 sm:grid-cols-2 lg:h-full lg:max-h-none"
      >
        <AnimatePresence initial={false}>
          {all.map((recording, index) => {
            const selected = recording.recording_id === selectedId;
            const upload = recording.recording_id.startsWith(UPLOAD_ID_PREFIX);
            const url = audioUrlFor(recording.recording_id);
            const playing = playingUrl === url;
            return (
              <motion.li
                layout
                key={recording.recording_id}
                initial={{ opacity: 0, y: 16 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true }}
                exit={{ opacity: 0, scale: 0.95 }}
                animate={{ scale: selected ? 1.02 : 1 }}
                transition={{ delay: (index % 9) * 0.03, scale: { type: "spring", stiffness: 300, damping: 22 } }}
              >
                <div
                  className={`dz-card relative flex h-full flex-col gap-3 rounded-2xl p-4 transition-shadow ${
                    selected ? "dz-selected" : ""
                  }`}
                >
                  <div className="flex items-start gap-3">
                    <button
                      type="button"
                      onClick={() => togglePreview(url)}
                      aria-label={`${playing ? "Stop preview of" : "Preview"} ${recording.display_filename}`}
                      className="relative grid h-10 w-10 shrink-0 place-items-center rounded-full bg-white/5 text-white ring-1 ring-white/10 transition hover:scale-110 hover:bg-white/10"
                    >
                      {playing ? <Pause className="h-4 w-4" /> : <Play className="ml-0.5 h-4 w-4" />}
                    </button>
                    <div className="min-w-0 flex-1">
                      <div className="truncate font-mono text-sm text-white" title={recording.display_filename}>
                        {recording.display_filename}
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-slate-400">
                        <span
                          className={`rounded-full px-2 py-px font-semibold ring-1 ${
                            upload ? "dz-accent-text ring-white/20" : "text-slate-300 ring-white/10"
                          }`}
                        >
                          {upload ? "Your upload" : "AMI meeting"}
                        </span>
                        <span className="font-mono">{formatBytes(recording.size_bytes)}</span>
                      </div>
                    </div>
                  </div>


                  <button
                    type="button"
                    onClick={() => onSelect(recording.recording_id)}
                    disabled={disabled}
                    aria-pressed={selected}
                    className={`mt-auto inline-flex items-center justify-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-50 ${
                      selected ? "dz-primary-btn" : "bg-white/5 text-slate-200 ring-1 ring-white/10 hover:bg-white/10"
                    }`}
                  >
                    {selected ? (
                      <>
                        <Check className="h-3.5 w-3.5" /> Selected
                      </>
                    ) : (
                      "Select"
                    )}
                  </button>
                </div>
              </motion.li>
            );
          })}
        </AnimatePresence>
        {all.length === 0 && (
          <li className="flex h-full min-h-[168px] flex-col items-center justify-center gap-2 rounded-2xl p-4 text-center ring-1 ring-white/10 sm:col-span-2">
            {isLoading ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin text-slate-400" />
                <div className="text-sm text-slate-400">Loading recordings…</div>
              </>
            ) : (
              <>
                <span className="grid h-10 w-10 place-items-center rounded-full bg-white/5 ring-1 ring-white/10">
                  <FolderOpen className="h-4 w-4 text-slate-400" />
                </span>
                <div className="font-display text-sm font-semibold text-white">No recordings available</div>
                <div className="text-xs text-slate-400">
                  The demo dataset has no recordings and you haven't uploaded any yet. Upload your own to get started.
                </div>
              </>
            )}
          </li>
        )}
      </motion.ul>
      <div className="lg:h-full lg:p-1">
        <UploadCard onUpload={onUpload} isUploading={isUploading} disabled={disabled} />
      </div>
    </div>
  );
};

const UploadCard = ({
  onUpload,
  isUploading,
  disabled,
}: {
  onUpload: (file: File) => void;
  isUploading: boolean;
  disabled: boolean;
}) => {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [rejection, setRejection] = useState<string | null>(null);
  const blocked = disabled || isUploading;

  const accept = (file: File | undefined) => {
    if (!file || blocked) return;
    const extension = file.name.split(".").pop()?.toLowerCase() ?? "";
    if (!ALLOWED_EXTENSIONS.includes(extension)) {
      setRejection("That file type isn't supported. Use WAV, MP3, M4A or FLAC.");
      return;
    }
    if (file.size > MAX_FILE_BYTES) {
      setRejection(`That file is ${formatBytes(file.size)}; the limit is 50 MB.`);
      return;
    }
    setRejection(null);
    onUpload(file);
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setDragging(false);
    accept(event.dataTransfer.files?.[0]);
  };

  return (
    <div
      role="button"
      tabIndex={blocked ? -1 : 0}
      aria-disabled={blocked}
      aria-label="Upload your own recording"
      data-dragging={dragging}
      onClick={() => !blocked && input.current?.click()}
      onKeyDown={(event) => {
        if (!blocked && (event.key === "Enter" || event.key === " ")) {
          event.preventDefault();
          input.current?.click();
        }
      }}
      onDragOver={(event) => {
        event.preventDefault();
        if (!blocked) setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      className={`dz-dropzone flex h-full min-h-[168px] flex-col items-center justify-center gap-2 rounded-2xl p-4 text-center transition ${
        blocked ? "cursor-not-allowed opacity-60" : "cursor-pointer hover:bg-white/5"
      }`}
    >
      <span className="grid h-10 w-10 place-items-center rounded-full bg-white/5 ring-1 ring-white/10">
        {isUploading ? <Loader2 className="h-4 w-4 animate-spin dz-accent-text" /> : <Upload className="h-4 w-4 dz-accent-text" />}
      </span>
      <div className="font-display text-sm font-semibold text-white">
        {isUploading ? "Uploading…" : "Upload your own"}
      </div>
      <div className="text-xs text-slate-400">Click or drop a file · WAV, MP3, M4A or FLAC · up to 50 MB</div>
      <div className="text-[11px] text-slate-500">Private to your session.</div>
      {rejection && <div className="text-xs text-rose-200">{rejection}</div>}
      <input
        ref={input}
        type="file"
        accept="audio/wav,audio/mpeg,audio/mp4,audio/flac,.wav,.mp3,.m4a,.flac"
        className="hidden"
        onChange={(event) => {
          accept(event.target.files?.[0]);
          // Reset so re-picking the same file fires onChange again.
          event.target.value = "";
        }}
      />
    </div>
  );
};

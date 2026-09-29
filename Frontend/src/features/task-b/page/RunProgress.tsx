import { useEffect, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { Check } from "lucide-react";

const STAGES = [
  { title: "Finding where people talk", text: "Voice activity: speech vs silence" },
  { title: "Listening to each voice", text: "A voice fingerprint per segment" },
  { title: "Grouping similar voices", text: "Clustering fingerprints into speakers" },
];
/** Seconds after which each stage lights up. The backend doesn't stream
 *  progress, so these are a timer — the page says so. The last one holds. */
const STAGE_STARTS = [0, 4, 15];
/** A cached re-run answers in milliseconds; don't flash the stages for it. */
const SHOW_AFTER_MS = 400;

const clock = (seconds: number) =>
  `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;

/**
 * The long-run indicator. A first run on a full meeting takes minutes on CPU,
 * so instead of a bare spinner it lights up the pipeline's three stages in
 * turn and shows the elapsed time.
 */
export const RunProgress = ({ running }: { running: boolean }) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!running) {
      setElapsed(0);
      return;
    }
    const started = Date.now();
    const timer = window.setInterval(() => setElapsed((Date.now() - started) / 1000), 250);
    return () => window.clearInterval(timer);
  }, [running]);

  const visible = running && elapsed * 1000 >= SHOW_AFTER_MS;
  const stage = STAGE_STARTS.reduce((current, start, index) => (elapsed >= start ? index : current), 0);

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          key="progress"
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          className="dz-card rounded-2xl p-5"
          role="status"
          aria-live="polite"
        >
          <ol className="grid gap-3 sm:grid-cols-3">
            {STAGES.map((item, index) => {
              const state = index < stage ? "done" : index === stage ? "active" : "waiting";
              return (
                <li
                  key={item.title}
                  className={`flex items-start gap-3 rounded-xl p-3 transition-opacity ${
                    state === "waiting" ? "opacity-40" : "opacity-100"
                  } ${state === "active" ? "bg-white/5 ring-1 ring-white/10" : ""}`}
                  aria-current={state === "active" ? "step" : undefined}
                >
                  <StageIcon index={index} state={state} />
                  <div>
                    <div className="text-sm font-semibold text-white">{item.title}</div>
                    <div className="text-xs text-slate-400">{item.text}</div>
                  </div>
                </li>
              );
            })}
          </ol>
          <div className="mt-4 flex flex-wrap items-center justify-between gap-2 text-xs text-slate-400">
            <span>
              The first run on a full meeting can take a few minutes; re-runs are instant. Stages advance on a timer —
              the model doesn&apos;t report progress.
            </span>
            <span className="font-mono text-slate-300">{clock(elapsed)}</span>
          </div>
        </motion.div>
      )}
    </AnimatePresence>
  );
};

const StageIcon = ({ index, state }: { index: number; state: "done" | "active" | "waiting" }) => {
  if (state === "done") {
    return (
      <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full dz-primary-btn">
        <Check className="h-4 w-4" />
      </span>
    );
  }
  const active = state === "active";
  return (
    <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-white/5 ring-1 ring-white/10">
      <svg viewBox="0 0 20 20" className="h-5 w-5 dz-accent-text" aria-hidden>
        {index === 0 &&
          // voice activity: bouncing level bars
          [3, 8, 13].map((x, bar) => (
            <rect
              key={x}
              x={x}
              y={4}
              width={3}
              height={12}
              rx={1.5}
              fill="currentColor"
              className={active ? "dz-bounce" : undefined}
              style={{ animationDelay: `${bar * 0.15}s`, transformBox: "fill-box" }}
            />
          ))}
        {index === 1 && (
          // listening: a breathing ring around a dot
          <>
            <circle cx={10} cy={10} r={3} fill="currentColor" />
            <circle cx={10} cy={10} r={7} fill="none" stroke="currentColor" strokeWidth={1.5} className={active ? "dz-breathe" : undefined} />
          </>
        )}
        {index === 2 &&
          // grouping: three dots pulling together
          [
            [5, 6],
            [15, 7],
            [10, 15],
          ].map(([cx, cy], dot) => (
            <circle
              key={dot}
              cx={cx}
              cy={cy}
              r={2.6}
              fill="currentColor"
              className={active ? "dz-breathe" : undefined}
              style={{ animationDelay: `${dot * 0.2}s` }}
            />
          ))}
      </svg>
    </span>
  );
};

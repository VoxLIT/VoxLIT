import { ReactNode, useId, useState } from "react";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, ChevronDown, Microscope } from "lucide-react";
import { usePalette } from "./theme";

/** Second-level information: hidden until asked for, then eased open. */
export const Disclosure = ({
  title = "Technical details",
  children,
  defaultOpen = false,
}: {
  title?: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const id = useId();
  return (
    <div className="rounded-sm border border-border bg-white/[0.02]">
      <button
        type="button"
        aria-expanded={open}
        aria-controls={id}
        onClick={() => setOpen((value) => !value)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-xs font-medium text-slate-300 transition-colors hover:text-white"
      >
        <Microscope className="h-3.5 w-3.5 text-violet-300" />
        <span className="flex-1">{title}</span>
        <motion.span animate={{ rotate: open ? 180 : 0 }} transition={{ type: "spring", stiffness: 300, damping: 22 }}>
          <ChevronDown className="h-4 w-4" />
        </motion.span>
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            id={id}
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.3, ease: [0.2, 0.7, 0.3, 1] }}
            className="overflow-hidden"
          >
            <div className="space-y-2 border-t border-border px-3 py-2.5 text-xs text-slate-300">{children}</div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export const SectionTitle = ({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: ReactNode;
  children?: ReactNode;
}) => (
  <motion.div
    initial={{ opacity: 0, y: 24 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true, margin: "-80px" }}
    transition={{ duration: 0.6, ease: [0.2, 0.7, 0.3, 1] }}
    className="mb-3 max-w-4xl"
  >
    <div className="text-[10px] font-semibold uppercase tracking-wider text-cyan-300/80">{eyebrow}</div>
    <h2 className="font-display text-base font-bold text-white">{title}</h2>
    {children && <p className="mt-1 text-xs text-slate-400">{children}</p>}
  </motion.div>
);

/** The "what this does" picture shown before a feature is run. */
export const FeatureImage = ({
  src,
  alt,
  caption,
  className = "",
}: {
  src: string;
  alt: string;
  caption?: string;
  className?: string;
}) => (
  <figure
    className={`group overflow-hidden rounded-sm border border-border ${className.includes("absolute") ? "" : "relative"} ${className}`}
  >
    <motion.img
      src={src}
      alt={alt}
      loading="lazy"
      className="h-full w-full object-cover"
      initial={{ scale: 1.08 }}
      whileInView={{ scale: 1 }}
      whileHover={{ scale: 1.05 }}
      viewport={{ once: true }}
      transition={{ duration: 1.2, ease: [0.2, 0.7, 0.3, 1] }}
    />
    <div className="pointer-events-none df-image-fade absolute inset-0" />
    {caption && (
      <figcaption className="absolute inset-x-0 bottom-0 p-3 text-xs font-medium text-slate-200">{caption}</figcaption>
    )}
  </figure>
);

export const VerdictChip = ({ spoof, size = "md" }: { spoof: boolean; size?: "sm" | "md" }) => {
  const { REAL, FAKE } = usePalette();
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-sm font-semibold ${
        size === "sm" ? "px-1.5 py-0.5 text-[11px]" : "px-2 py-0.5 text-xs"
      }`}
      style={{
        color: spoof ? FAKE : REAL,
        background: `${spoof ? FAKE : REAL}1f`,
        boxShadow: `inset 0 0 0 1px ${spoof ? FAKE : REAL}59`,
      }}
    >
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: spoof ? FAKE : REAL }} />
      {spoof ? "Sounds synthetic" : "Sounds real"}
    </span>
  );
};

export const ErrorNote = ({ children }: { children: ReactNode }) => (
  <motion.div
    role="alert"
    initial={{ opacity: 0, y: -6 }}
    animate={{ opacity: 1, y: 0 }}
    className="flex items-start gap-2 rounded-sm border border-rose-400/30 bg-rose-500/10 p-2.5 text-xs text-rose-200"
  >
    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
    <span>{children}</span>
  </motion.div>
);

/** A friendly finding: the headline a non-expert reads first. */
export const Finding = ({ alarming, title, detail }: { alarming: boolean; title: string; detail: string }) => (
  <motion.div
    initial={{ opacity: 0, y: 10 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ delay: 0.2 }}
    className={`rounded-sm border p-3 ${
      alarming ? "border-amber-300/40 bg-amber-400/10" : "border-emerald-300/30 bg-emerald-400/10"
    }`}
  >
    <div className={`text-sm font-bold ${alarming ? "text-amber-200" : "text-emerald-200"}`}>{title}</div>
    <p className="mt-1 text-xs text-slate-300">{detail}</p>
  </motion.div>
);

export const PrimaryButton = ({
  children,
  onClick,
  disabled,
  busy,
  className = "",
}: {
  children: ReactNode;
  onClick: () => void;
  disabled?: boolean;
  busy?: boolean;
  className?: string;
}) => (
  <motion.button
    type="button"
    onClick={onClick}
    disabled={disabled || busy}
    whileTap={disabled || busy ? undefined : { scale: 0.98 }}
    className={`relative inline-flex items-center justify-center gap-1.5 overflow-hidden df-primary-btn px-3 py-1.5 text-xs font-semibold transition-opacity disabled:cursor-not-allowed disabled:opacity-50 ${className}`}
  >
    {busy && <span className="df-scan absolute inset-y-0 left-0 w-1/4 bg-white/40 blur-md" aria-hidden />}
    <span className="relative inline-flex items-center gap-1.5">{children}</span>
  </motion.button>
);

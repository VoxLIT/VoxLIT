import { createContext, ReactNode, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { AnimatePresence, motion } from "motion/react";
import { Moon, Sun } from "lucide-react";
import { DARK_PALETTE, LIGHT_PALETTE, type Palette } from "./palette";

export type DzTheme = "dark" | "light";

const STORAGE_KEY = "voxlit:diarization:theme";

interface ThemeValue {
  theme: DzTheme;
  palette: Palette;
  /** Switch themes; `origin` (client px) is where the reveal circle grows from. */
  toggle: (origin?: { x: number; y: number }) => void;
}

const ThemeContext = createContext<ThemeValue>({
  theme: "dark",
  palette: DARK_PALETTE,
  toggle: () => undefined,
});

const readInitialTheme = (): DzTheme => {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved === "dark" || saved === "light") return saved;
  } catch {
    // storage blocked (private mode) — fall through to the system preference
  }
  return window.matchMedia?.("(prefers-color-scheme: light)").matches ? "light" : "dark";
};

type ViewTransitionDocument = Document & {
  startViewTransition?: (update: () => void) => { ready: Promise<void> };
};

/**
 * The diarization page's own light/dark switch. It is local to this page (the
 * other task pages keep the shared theme) and remembered per browser.
 *
 * Where the browser supports the View Transitions API the new theme is
 * revealed as a circle growing from the toggle; elsewhere it just swaps.
 */
export const DzThemeProvider = ({ children }: { children: (theme: DzTheme) => ReactNode }) => {
  const [theme, setTheme] = useState<DzTheme>(readInitialTheme);

  useEffect(() => {
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // not remembered, but still applied
    }
  }, [theme]);

  const toggle = useCallback((origin?: { x: number; y: number }) => {
    const next = (current: DzTheme): DzTheme => (current === "dark" ? "light" : "dark");
    const doc = document as ViewTransitionDocument;
    const reduced = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (!doc.startViewTransition || reduced || !origin) {
      setTheme(next);
      return;
    }
    // The browser snapshots the DOM when the callback returns, so the new
    // theme has to be committed synchronously inside it.
    const transition = doc.startViewTransition(() => flushSync(() => setTheme(next)));
    const radius = Math.hypot(Math.max(origin.x, innerWidth - origin.x), Math.max(origin.y, innerHeight - origin.y));
    transition.ready
      .then(() => {
        document.documentElement.animate(
          { clipPath: [`circle(0px at ${origin.x}px ${origin.y}px)`, `circle(${radius}px at ${origin.x}px ${origin.y}px)`] },
          { duration: 650, easing: "cubic-bezier(0.2, 0.7, 0.3, 1)", pseudoElement: "::view-transition-new(root)" },
        );
      })
      .catch(() => undefined);
  }, []);

  const value = useMemo(
    () => ({ theme, palette: theme === "light" ? LIGHT_PALETTE : DARK_PALETTE, toggle }),
    [theme, toggle],
  );
  return <ThemeContext.Provider value={value}>{children(theme)}</ThemeContext.Provider>;
};

export const useDzTheme = () => useContext(ThemeContext);
/** The colours for the active theme. React context does not cross an r3f
 *  `<Canvas>`, so read this outside one and pass the colours in as props. */
export const usePalette = () => useContext(ThemeContext).palette;

/** Sun/moon switch; the icons roll past each other when it flips. */
export const ThemeToggle = () => {
  const { theme, toggle } = useDzTheme();
  const button = useRef<HTMLButtonElement>(null);
  const light = theme === "light";
  return (
    <button
      ref={button}
      type="button"
      role="switch"
      aria-checked={light}
      aria-label="Light theme"
      title={light ? "Switch to dark theme" : "Switch to light theme"}
      onClick={() => {
        const box = button.current?.getBoundingClientRect();
        toggle(box ? { x: box.left + box.width / 2, y: box.top + box.height / 2 } : undefined);
      }}
      className="relative grid h-9 w-9 place-items-center overflow-hidden rounded-full ring-1 ring-white/10 transition hover:bg-white/10"
    >
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={theme}
          initial={{ y: 18, rotate: -90, opacity: 0 }}
          animate={{ y: 0, rotate: 0, opacity: 1 }}
          exit={{ y: -18, rotate: 90, opacity: 0 }}
          transition={{ type: "spring", stiffness: 300, damping: 20 }}
          className="grid place-items-center"
        >
          {light ? <Sun className="h-4 w-4 text-amber-500" /> : <Moon className="h-4 w-4 dz-accent-text" />}
        </motion.span>
      </AnimatePresence>
    </button>
  );
};

import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, ChevronDown, Github, Menu, X } from "lucide-react";
import { TASKS } from "@/tasks/registry";
import { GITHUB_URL, TASK_SHOWCASE } from "./siteContent";
import TaskIllustration from "./TaskIllustration";

const SECTION_LINKS = [
  { href: "#workbench", label: "How it works" },
  { href: "#about", label: "About" },
  { href: "#contact", label: "Contact" },
];

/** Floating pill navigation bar for the landing page. */
const SiteNav = () => {
  const [tasksOpen, setTasksOpen] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [scrolled, setScrolled] = useState(false);
  const navRef = useRef<HTMLElement>(null);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  // Close menus on outside click or Escape
  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (navRef.current && !navRef.current.contains(e.target as Node)) {
        setTasksOpen(false);
        setMobileOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setTasksOpen(false);
        setMobileOpen(false);
      }
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, []);

  const closeAll = () => {
    setTasksOpen(false);
    setMobileOpen(false);
  };

  return (
    <nav ref={navRef} className="fixed inset-x-0 top-3 z-50 px-4" aria-label="Main">
      <div
        className={`relative mx-auto flex h-14 max-w-6xl items-center justify-between gap-4 rounded-full border border-border pl-5 pr-2 backdrop-blur-xl transition-all duration-300 ${
          scrolled ? "bg-white/90 shadow-aws-md" : "bg-white/70 shadow-aws-sm"
        }`}
      >
        <a href="#top" onClick={closeAll} className="flex shrink-0 items-center" aria-label="VoxLIT home">
          <img src="/home/logo-light.png" alt="VoxLIT" className="h-6 w-auto" />
        </a>

        {/* Desktop links */}
        <div className="hidden md:flex items-center gap-1 text-[13px] font-medium text-slate-600">
          <div className="relative" onMouseEnter={() => setTasksOpen(true)} onMouseLeave={() => setTasksOpen(false)}>
            <button
              type="button"
              onClick={() => setTasksOpen((o) => !o)}
              aria-expanded={tasksOpen}
              className="flex items-center gap-1 rounded-full px-3.5 py-2 transition-colors hover:bg-slate-100 hover:text-foreground"
            >
              Tasks
              <ChevronDown className={`h-3.5 w-3.5 transition-transform ${tasksOpen ? "rotate-180" : ""}`} />
            </button>
            {/* pt-3 keeps the hover bridge between trigger and panel */}
            <div
              className={`absolute left-1/2 top-full w-[26rem] -translate-x-1/2 pt-3 transition-all duration-200 ${
                tasksOpen ? "visible opacity-100 translate-y-0" : "invisible opacity-0 -translate-y-1"
              }`}
            >
              <div className="rounded-2xl border border-border bg-white p-2 shadow-aws-xl">
                {TASKS.map((task) => (
                  <Link
                    key={task.id}
                    to={task.route}
                    onClick={closeAll}
                    className="group flex items-center gap-3 rounded-xl p-2 transition-colors hover:bg-slate-50"
                  >
                    <div className="h-10 w-16 shrink-0 overflow-hidden rounded-lg border border-border bg-slate-50">
                      <TaskIllustration taskId={task.id} className="h-full w-full" />
                    </div>
                    <div className="min-w-0">
                      <div className="text-[13px] font-medium text-foreground">{task.name}</div>
                      <div className="truncate text-xs text-muted-foreground">{TASK_SHOWCASE[task.id].tagline}</div>
                    </div>
                    <ArrowRight className="ml-auto h-3.5 w-3.5 shrink-0 text-transparent transition-colors group-hover:text-primary" />
                  </Link>
                ))}
              </div>
            </div>
          </div>
          {SECTION_LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              onClick={closeAll}
              className="rounded-full px-3.5 py-2 transition-colors hover:bg-slate-100 hover:text-foreground"
            >
              {l.label}
            </a>
          ))}
        </div>

        <div className="flex items-center gap-1.5">
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            aria-label="VoxLIT on GitHub"
            className="hidden sm:flex h-10 w-10 items-center justify-center rounded-full text-slate-600 transition-colors hover:bg-slate-100 hover:text-foreground"
          >
            <Github className="h-4 w-4" />
          </a>
          <a
            href="#tasks"
            onClick={closeAll}
            className="hidden sm:inline-flex h-10 items-center gap-1.5 rounded-full bg-primary px-5 text-[13px] font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
          >
            Get started
            <ArrowRight className="h-3.5 w-3.5" />
          </a>
          <button
            type="button"
            onClick={() => setMobileOpen((o) => !o)}
            aria-expanded={mobileOpen}
            aria-label="Toggle menu"
            className="md:hidden flex h-10 w-10 items-center justify-center rounded-full text-foreground hover:bg-slate-100"
          >
            {mobileOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile panel */}
      {mobileOpen && (
        <div className="md:hidden mx-auto mt-2 max-w-6xl rounded-2xl border border-border bg-white p-3 shadow-aws-xl">
          <div className="px-2 pb-1 pt-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground">Tasks</div>
          {TASKS.map((task) => (
            <Link
              key={task.id}
              to={task.route}
              onClick={closeAll}
              className="flex items-center gap-3 rounded-xl p-2 text-sm text-foreground hover:bg-slate-50"
            >
              <div className="h-8 w-12 shrink-0 overflow-hidden rounded-md border border-border bg-slate-50">
                <TaskIllustration taskId={task.id} className="h-full w-full" />
              </div>
              {task.name}
            </Link>
          ))}
          <div className="my-2 h-px bg-border" />
          {SECTION_LINKS.map((l) => (
            <a key={l.href} href={l.href} onClick={closeAll} className="block rounded-xl px-2 py-2 text-sm text-slate-600 hover:bg-slate-50">
              {l.label}
            </a>
          ))}
          <a href={GITHUB_URL} target="_blank" rel="noreferrer" className="flex items-center gap-2 rounded-xl px-2 py-2 text-sm text-slate-600 hover:bg-slate-50">
            <Github className="h-4 w-4" /> GitHub
          </a>
        </div>
      )}
    </nav>
  );
};

export default SiteNav;

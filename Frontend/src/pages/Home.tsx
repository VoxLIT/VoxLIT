import { ReactNode, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  ArrowRight,
  ArrowUpRight,
  Bug,
  GitPullRequest,
  Github,
  MessageSquare,
  Scale,
} from "lucide-react";
import { TASKS } from "@/tasks/registry";
import { TaskDefinition } from "@/tasks/types";
import SiteNav from "@/components/site/SiteNav";
import SiteFooter from "@/components/site/SiteFooter";
import QuickTranscribe from "@/components/site/QuickTranscribe";
import TaskIllustration from "@/components/site/TaskIllustration";
import WorkbenchDiagram from "@/components/site/WorkbenchDiagram";
import { GITHUB_URL, LIT_URL, TASK_SHOWCASE } from "@/components/site/siteContent";

/** Fades and rises children into view the first time they are scrolled to. */
const Reveal = ({ children, delay = 0, className = "" }: { children: ReactNode; delay?: number; className?: string }) => {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) setVisible(true);
      },
      { threshold: 0.15 }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      style={{ transitionDelay: `${delay}ms` }}
      className={`transition-all duration-700 ease-out ${
        visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-6"
      } ${className}`}
    >
      {children}
    </div>
  );
};

/** Animated bars in the logo's colours — the two tall ones in saliency orange. */
const EqualizerBars = () => {
  const heights = Array.from({ length: 28 }, (_, i) => 12 + ((i * 37) % 5) * 7 + (i % 3) * 4);
  return (
    <div className="flex h-12 items-end gap-1" aria-hidden="true">
      {heights.map((h, i) => (
        <div
          key={i}
          className={`eq-bar w-1 rounded-full ${i % 9 === 4 ? "bg-[#F06638]" : "bg-primary/35"}`}
          style={{ height: `${h}px`, animationDelay: `${i * 0.07}s` }}
        />
      ))}
    </div>
  );
};

const SectionHeading = ({ eyebrow, title, children }: { eyebrow: string; title: string; children?: ReactNode }) => (
  <div className="max-w-2xl space-y-2">
    <div className="text-[11px] font-semibold uppercase tracking-[0.14em] text-primary">{eyebrow}</div>
    <h2 className="text-2xl font-semibold tracking-tight text-foreground sm:text-[1.75rem]">{title}</h2>
    {children && <p className="text-sm leading-relaxed text-muted-foreground">{children}</p>}
  </div>
);

/** A clickable figure card that opens the task's workbench. */
const TaskTile = ({ task }: { task: TaskDefinition }) => {
  const show = TASK_SHOWCASE[task.id];

  return (
    <Link
      to={task.route}
      className="group flex h-full flex-col overflow-hidden rounded-2xl border border-border bg-white outline-none ring-primary ring-offset-2 transition-all duration-300 hover:-translate-y-1 hover:border-primary/30 hover:shadow-aws-lg focus-visible:ring-2"
    >
      <div className="relative border-b border-border bg-[linear-gradient(to_bottom,#F8FAFC,#FFFFFF)]">
        <div
          aria-hidden
          className="absolute inset-0 opacity-60 [background-image:radial-gradient(#E2E8F0_1px,transparent_1px)] [background-size:14px_14px]"
        />
        <TaskIllustration
          taskId={task.id}
          className="relative block aspect-[400/216] w-full transition-transform duration-500 ease-out group-hover:scale-[1.03]"
        />
      </div>
      <div className="flex flex-1 flex-col gap-2 p-5">
        <div className="flex items-start justify-between gap-3">
          <h3 className="text-base font-semibold text-foreground">{task.name}</h3>
          {task.status !== "active" && (
            <span className="shrink-0 rounded-full bg-muted px-2 py-0.5 text-[10px] text-muted-foreground">Coming soon</span>
          )}
        </div>
        <p className="text-[13px] leading-snug text-muted-foreground">{show.tagline}</p>
        <span className="mt-auto inline-flex items-center gap-1.5 pt-3 text-[13px] font-medium text-primary">
          Open workbench
          <ArrowRight className="h-3.5 w-3.5 transition-transform group-hover:translate-x-0.5" />
        </span>
      </div>
    </Link>
  );
};

const STEPS = [
  {
    title: "Choose a task and model",
    body: "Each workbench loads its own models and datasets, or a clip you upload.",
  },
  {
    title: "Select a clip",
    body: "Click a point in the embedding plot or a row in the dataset table.",
  },
  {
    title: "Read the explanation",
    body: "Saliency, attention and perturbation views update for that clip, beside its prediction and metadata.",
  },
];

const Home = () => {
  const modelCount = new Set(TASKS.flatMap((t) => t.models.filter((m) => m.available).map((m) => m.id))).size;
  const datasetCount = new Set(TASKS.flatMap((t) => t.datasets.filter((d) => d.available).map((d) => d.id))).size;

  const scrollToTasks = () => document.getElementById("tasks")?.scrollIntoView({ behavior: "smooth" });

  return (
    <div id="top" className="min-h-screen bg-white flex flex-col">
      <SiteNav />

      <main className="flex-1">
        {/* Hero */}
        <section className="relative overflow-hidden border-b border-border bg-white pb-16 pt-32 sm:pt-36">
          <div
            aria-hidden
            className="absolute inset-0 opacity-70 [background-image:linear-gradient(to_right,#EEF2F7_1px,transparent_1px),linear-gradient(to_bottom,#EEF2F7_1px,transparent_1px)] [background-size:44px_44px] [mask-image:radial-gradient(ellipse_at_top,black,transparent_70%)]"
          />
          <div aria-hidden className="absolute -top-40 left-1/2 h-80 w-[48rem] -translate-x-1/2 rounded-full bg-primary/10 blur-3xl" />

          <div className="relative mx-auto max-w-6xl px-5">
            <Reveal className="mx-auto max-w-3xl space-y-5 text-center">
              <h1 className="text-3xl font-semibold leading-tight tracking-tight text-slate-900 sm:text-4xl lg:text-[2.75rem]">
                Learning Interpretability Tool for Voice Models
              </h1>
              <p className="mx-auto max-w-2xl text-base leading-relaxed text-slate-600">
                An interactive workbench for probing speech models one clip at a time: saliency,
                attention, embeddings and robustness, across five tasks. Built on the approach of{" "}
                <a
                  href={LIT_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium text-slate-800 underline decoration-slate-300 underline-offset-4 hover:text-primary hover:decoration-primary"
                >
                  Google&apos;s LIT
                </a>
                .
              </p>
            </Reveal>

            <div id="try" className="mt-10 scroll-mt-28">
              <Reveal delay={120}>
                <QuickTranscribe onContinue={scrollToTasks} />
              </Reveal>
            </div>

            <Reveal delay={200} className="mx-auto mt-14 flex max-w-3xl flex-wrap items-center justify-center gap-x-12 gap-y-6">
              {[
                [TASKS.length, "Tasks"],
                [modelCount, "Models"],
                [datasetCount, "Datasets"],
              ].map(([n, label]) => (
                <div key={label as string} className="text-center">
                  <div className="text-3xl font-semibold tabular-nums text-slate-900">{n}</div>
                  <div className="text-[11px] uppercase tracking-wider text-muted-foreground">{label}</div>
                </div>
              ))}
              <EqualizerBars />
              <button
                type="button"
                onClick={scrollToTasks}
                className="inline-flex items-center gap-1.5 text-sm font-medium text-primary hover:underline"
              >
                Browse tasks <ArrowRight className="h-4 w-4" />
              </button>
            </Reveal>
          </div>
        </section>

        {/* Tasks */}
        <section id="tasks" className="bg-white px-5 py-20 scroll-mt-20">
          <div className="mx-auto max-w-6xl space-y-10">
            <div className="flex flex-wrap items-end justify-between gap-4">
              <SectionHeading eyebrow="Analysis tasks" title="Pick a task to open its workbench">
                Each one has its own models, data and explanation tools.
              </SectionHeading>
            </div>
            <div className="grid gap-5 sm:grid-cols-2 lg:grid-cols-6">
              {TASKS.map((task, i) => (
                <Reveal
                  key={task.id}
                  delay={(i % 3) * 90}
                  className={`h-full lg:col-span-2 ${TASKS.length % 3 === 2 && i === TASKS.length - 2 ? "lg:col-start-2" : ""}`}
                >
                  <TaskTile task={task} />
                </Reveal>
              ))}
            </div>
          </div>
        </section>

        {/* How it works */}
        <section id="workbench" className="border-y border-border bg-slate-50/70 px-5 py-20 scroll-mt-20">
          <div className="mx-auto grid max-w-6xl items-center gap-12 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="space-y-8">
              <SectionHeading eyebrow="How it works" title="Three linked panels, one datapoint at a time">
                Every workbench shares the same layout, so what you learn on one task carries over to
                the rest.
              </SectionHeading>
              <ol className="relative space-y-6">
                <div aria-hidden className="absolute bottom-4 left-[15px] top-4 w-px bg-border" />
                {STEPS.map(({ title, body }, i) => (
                  <Reveal key={title} delay={i * 100}>
                    <li className="relative flex gap-4">
                      <span className="relative z-10 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary/30 bg-white text-xs font-semibold text-primary">
                        {i + 1}
                      </span>
                      <div className="pt-1">
                        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
                        <p className="mt-1 text-[13px] leading-relaxed text-muted-foreground">{body}</p>
                      </div>
                    </li>
                  </Reveal>
                ))}
              </ol>
              <button
                type="button"
                onClick={scrollToTasks}
                className="inline-flex h-10 items-center gap-2 rounded-full bg-primary px-5 text-sm font-medium text-primary-foreground transition-colors hover:bg-primary-hover"
              >
                Open a workbench <ArrowRight className="h-4 w-4" />
              </button>
            </div>
            <Reveal delay={150}>
              <div className="rounded-2xl bg-white p-2 shadow-aws-lg ring-1 ring-border">
                <WorkbenchDiagram className="block h-auto w-full" />
              </div>
              <p className="mt-3 text-center text-xs text-muted-foreground">
                Selecting a clip in the embedding plot drives the other two panels.
              </p>
            </Reveal>
          </div>
        </section>

        {/* About */}
        <section id="about" className="bg-white px-5 py-20 scroll-mt-20">
          <Reveal className="mx-auto max-w-6xl">
            <div className="grid items-center gap-8 rounded-2xl border border-border bg-white p-8 shadow-aws-sm sm:p-10 md:grid-cols-[1.5fr_1fr]">
              <SectionHeading eyebrow="About VoxLIT" title="Built for researchers who need to know why, not just what">
                VoxLIT carries the interpretability approach of Google&apos;s LIT over to audio, so speech
                models can be questioned the same way text models are. It is open source, and each task
                is maintained as its own module.
              </SectionHeading>
              <div className="flex flex-col gap-3 md:items-end">
                <a
                  href={GITHUB_URL}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex h-11 items-center justify-center gap-2 rounded-full bg-slate-900 px-6 text-sm font-medium text-white transition-colors hover:bg-slate-800"
                >
                  <Github className="h-4 w-4" /> View the source
                </a>
                <span className="inline-flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Scale className="h-3.5 w-3.5" /> Open source · MIT License
                </span>
              </div>
            </div>
          </Reveal>
        </section>

        {/* Contact */}
        <section id="contact" className="border-t border-border bg-white px-5 py-20 scroll-mt-20">
          <div className="mx-auto max-w-6xl space-y-10">
            <SectionHeading eyebrow="Contact" title="Get in touch">
              Questions, bug reports or collaboration ideas are all welcome.
            </SectionHeading>
            <div className="grid gap-5 md:grid-cols-3">
              {[
                {
                  icon: MessageSquare,
                  title: "Ask a question",
                  body: "Start a thread on GitHub and the team will reply there.",
                  href: `${GITHUB_URL}/issues/new`,
                  cta: "Open an issue",
                },
                {
                  icon: Bug,
                  title: "Report a problem",
                  body: "Found a bug in a workbench? Tell us which task and what you saw.",
                  href: `${GITHUB_URL}/issues/new`,
                  cta: "Report a bug",
                },
                {
                  icon: GitPullRequest,
                  title: "Collaborate",
                  body: "Want to add a task or a model? The contributing guide shows how.",
                  href: `${GITHUB_URL}/blob/main/CONTRIBUTING.md`,
                  cta: "Read the guide",
                },
              ].map(({ icon: Icon, title, body, href, cta }, i) => (
                <Reveal key={title} delay={i * 90} className="h-full">
                  <a
                    href={href}
                    target="_blank"
                    rel="noreferrer"
                    className="group flex h-full flex-col gap-3 rounded-2xl border border-border bg-white p-6 transition-all hover:-translate-y-0.5 hover:border-primary/30 hover:shadow-aws-md"
                  >
                    <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-primary/10 text-primary">
                      <Icon className="h-5 w-5" />
                    </div>
                    <h3 className="text-base font-semibold text-foreground">{title}</h3>
                    <p className="text-[13px] leading-relaxed text-muted-foreground">{body}</p>
                    <span className="mt-auto inline-flex items-center gap-1 pt-2 text-[13px] font-medium text-primary">
                      {cta} <ArrowUpRight className="h-3.5 w-3.5" />
                    </span>
                  </a>
                </Reveal>
              ))}
            </div>
          </div>
        </section>
      </main>

      <SiteFooter />
    </div>
  );
};

export default Home;

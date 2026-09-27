import { ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowUp, ArrowUpRight, Bug, Github } from "lucide-react";
import { TASKS } from "@/tasks/registry";
import { GITHUB_URL, LIT_URL } from "./siteContent";

const FooterHeading = ({ children }: { children: string }) => (
  <h3 className="mb-4 text-xs font-semibold text-foreground">{children}</h3>
);

const linkClass = "text-[13px] text-slate-600 transition-colors hover:text-primary";

const External = ({ href, children }: { href: string; children: ReactNode }) => (
  <a href={href} target="_blank" rel="noreferrer" className={`${linkClass} group inline-flex items-center gap-1`}>
    {children}
    <ArrowUpRight className="h-3 w-3 opacity-50 transition-opacity group-hover:opacity-100" />
  </a>
);

const iconLink =
  "flex h-9 w-9 items-center justify-center rounded-full border border-border text-slate-600 transition-colors hover:border-primary/40 hover:text-primary";

const SiteFooter = () => (
  <footer className="border-t border-border bg-white">
    <div className="mx-auto max-w-6xl px-5 pb-8 pt-16">
      <div className="grid gap-10 sm:grid-cols-2 lg:grid-cols-12">
        <div className="space-y-5 sm:col-span-2 lg:col-span-4">
          <img src="/home/logo-light.png" alt="VoxLIT" className="h-7 w-auto" />
          <p className="max-w-xs text-[13px] leading-relaxed text-muted-foreground">
            An open-source interpretability workbench for speech models, extending Google&apos;s
            Learning Interpretability Tool from text to audio.
          </p>
          <div className="flex items-center gap-2">
            <a href={GITHUB_URL} target="_blank" rel="noreferrer" aria-label="GitHub repository" className={iconLink}>
              <Github className="h-4 w-4" />
            </a>
            <a href={`${GITHUB_URL}/issues`} target="_blank" rel="noreferrer" aria-label="Issue tracker" className={iconLink}>
              <Bug className="h-4 w-4" />
            </a>
          </div>
        </div>

        <div className="lg:col-span-3">
          <FooterHeading>Workbenches</FooterHeading>
          <ul className="space-y-2.5">
            {TASKS.map((t) => (
              <li key={t.id}>
                <Link to={t.route} className={linkClass}>
                  {t.name}
                </Link>
              </li>
            ))}
          </ul>
        </div>

        <div className="lg:col-span-2">
          <FooterHeading>Project</FooterHeading>
          <ul className="space-y-2.5">
            <li><a href="#workbench" className={linkClass}>How it works</a></li>
            <li><a href="#about" className={linkClass}>About</a></li>
            <li><a href="#contact" className={linkClass}>Contact</a></li>
          </ul>
        </div>

        <div className="lg:col-span-3">
          <FooterHeading>Resources</FooterHeading>
          <ul className="space-y-2.5">
            <li><External href={GITHUB_URL}>Source code</External></li>
            <li><External href={`${GITHUB_URL}/blob/main/CONTRIBUTING.md`}>Contributing guide</External></li>
            <li><External href={`${GITHUB_URL}/issues/new`}>Report an issue</External></li>
            <li><External href={LIT_URL}>Google LIT</External></li>
          </ul>
        </div>
      </div>

      <div className="mt-14 flex flex-col gap-4 border-t border-border pt-6 text-xs text-muted-foreground sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
          <span>© {new Date().getFullYear()} VoxLIT contributors</span>
          <a href={`${GITHUB_URL}/blob/main/LICENSE`} target="_blank" rel="noreferrer" className="hover:text-primary">
            MIT License
          </a>
          <a href={`${GITHUB_URL}/blob/main/CODE_OF_CONDUCT.md`} target="_blank" rel="noreferrer" className="hover:text-primary">
            Code of conduct
          </a>
          <a href={`${GITHUB_URL}/blob/main/SECURITY.md`} target="_blank" rel="noreferrer" className="hover:text-primary">
            Security
          </a>
        </div>
        <a href="#top" className="inline-flex items-center gap-1.5 hover:text-primary">
          Back to top <ArrowUp className="h-3.5 w-3.5" />
        </a>
      </div>
    </div>
  </footer>
);

export default SiteFooter;

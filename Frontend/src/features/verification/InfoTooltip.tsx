import type { ReactNode } from "react";
import { HelpCircle } from "lucide-react";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";

interface InfoTooltipProps {
  /** Short heading, shown bold on its own line. */
  title: string;
  /** Body text: one or two short sentences (or a few short paragraphs). */
  children: ReactNode;
  /** Set when the icon sits inside a clickable parent that must not fire. */
  stopClickPropagation?: boolean;
}

/**
 * Help-icon tooltip for the Speaker Verification panels. The shared
 * TooltipContent has no max width, so long help text would render as one
 * very wide line; this caps the width and lets the text wrap.
 */
export const InfoTooltip = ({ title, children, stopClickPropagation = false }: InfoTooltipProps) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <HelpCircle
          aria-label={`About ${title}`}
          className="h-3 w-3 shrink-0 text-muted-foreground hover:text-primary cursor-help transition-colors"
          onClick={stopClickPropagation ? (event) => event.stopPropagation() : undefined}
        />
      </TooltipTrigger>
      <TooltipContent
        side="left"
        collisionPadding={12}
        className="max-w-xs whitespace-normal break-words px-3 py-2.5 text-xs font-normal leading-relaxed"
      >
        <p className="mb-1 font-semibold text-popover-foreground">{title}</p>
        <div className="space-y-1.5">{children}</div>
      </TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

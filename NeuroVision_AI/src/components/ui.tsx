import { forwardRef, useId, type ButtonHTMLAttributes, type ReactNode } from "react";
import type { LucideIcon } from "lucide-react";

/*
 * Instrument-style building blocks. Hierarchy comes from spacing, hairlines
 * and the black optical windows, not from stacks of identical cards.
 */

interface SectionProps {
  title: string;
  description?: string;
  actions?: ReactNode;
  children: ReactNode;
  className?: string;
  /** "paper" gives the section a recessed paper background (used for control areas). */
  surface?: "none" | "paper";
}

/** A titled region of the page, exposed to screen readers as a labelled <section>. */
export function Section({ title, description, actions, children, className = "", surface = "none" }: SectionProps) {
  const headingId = useId();
  const surfaceClass = surface === "paper" ? "paper-well p-4 sm:p-5" : "";
  return (
    <section aria-labelledby={headingId} className={`flex flex-col ${surfaceClass} ${className}`}>
      <div className="mb-3 flex flex-wrap items-end justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          <h2 id={headingId} className="text-base font-bold text-ink">
            {title}
          </h2>
          {description && <p className="mt-0.5 max-w-prose text-sm text-graphite">{description}</p>}
        </div>
        {actions && <div className="flex flex-wrap items-center gap-2">{actions}</div>}
      </div>
      {children}
    </section>
  );
}

type ButtonVariant = "primary" | "secondary" | "quiet" | "danger";

const BUTTON_STYLES: Record<ButtonVariant, string> = {
  primary: "bg-cobalt text-white hover:bg-cobalt-deep font-bold shadow-[0_8px_24px_rgba(0,0,0,0.16)]",
  secondary: "bg-raised text-ink border border-rule hover:border-cobalt/60 font-semibold",
  quiet: "text-ink hover:bg-raised font-semibold",
  danger: "bg-raised text-live border border-live/40 hover:border-live font-semibold",
};

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  icon?: LucideIcon;
  size?: "sm" | "md" | "lg";
}

const BUTTON_SIZES = {
  sm: "h-8 px-3 text-sm gap-1.5",
  md: "h-10 px-4 text-sm gap-2",
  lg: "h-12 px-5 text-base gap-2",
};

// forwardRef lets parents move keyboard focus onto the button (e.g. when a dialog opens).
export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "secondary", icon: Icon, size = "md", className = "", children, ...rest },
  ref,
) {
  return (
    <button
      ref={ref}
      type="button"
      className={`inline-flex shrink-0 items-center justify-center rounded-lg transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${BUTTON_SIZES[size]} ${BUTTON_STYLES[variant]} ${className}`}
      {...rest}
    >
      {Icon && <Icon size={size === "lg" ? 18 : 16} strokeWidth={2.2} aria-hidden />}
      {children}
    </button>
  );
});

type LightTone = "live" | "ok" | "busy" | "off" | "error";

const LIGHT_COLORS: Record<LightTone, string> = {
  live: "bg-live",
  ok: "bg-ok",
  busy: "bg-caution",
  off: "bg-graphite/40",
  error: "bg-live",
};

/** A small indicator light with a text label (the label always carries the meaning). */
export function StatusLight({ tone, children }: { tone: LightTone; children: ReactNode }) {
  return (
    <span className="inline-flex items-center gap-2 whitespace-nowrap text-sm text-ink">
      <span className={`h-2 w-2 rounded-full ${LIGHT_COLORS[tone]} ${tone === "live" ? "live-dot" : ""}`} aria-hidden />
      {children}
    </span>
  );
}

interface SegmentedOption<T extends string | number> {
  value: T;
  label: string;
}

interface SegmentedControlProps<T extends string | number> {
  label: string;
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  isLabelVisible?: boolean;
}

/** Mutually exclusive options (a radio group drawn as a single switch-plate). */
export function SegmentedControl<T extends string | number>({ label, options, value, onChange, isLabelVisible = true }: SegmentedControlProps<T>) {
  const labelId = useId();
  return (
    <div>
      <div id={labelId} className={isLabelVisible ? "mb-2 text-sm text-ink" : "sr-only"}>
        {label}
      </div>
      <div role="radiogroup" aria-labelledby={labelId} className="inline-flex max-w-full flex-wrap rounded-lg border border-rule bg-chassis p-0.5">
        {options.map((option) => {
          const isSelected = option.value === value;
          return (
            <button
              key={String(option.value)}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => onChange(option.value)}
              className={`whitespace-nowrap rounded-md px-3 py-1.5 text-sm transition-colors ${isSelected ? "bg-raised font-bold text-cobalt shadow-sm" : "text-graphite hover:text-ink"}`}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

/** A caution note for limitations and hypothetical content. */
export function Note({ children }: { children: ReactNode }) {
  return <p className="border-l-2 border-caution bg-caution-soft px-3 py-2 text-sm text-caution">{children}</p>;
}

/** One label/value pair in a readout row (rendered inside a <dl>). */
export function Readout({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="min-w-0 px-4 py-3 first:pl-0" title={hint}>
      <dt className="text-xs text-graphite">{label}</dt>
      <dd className="mt-0.5 truncate font-mono text-base font-semibold text-ink">{value}</dd>
    </div>
  );
}

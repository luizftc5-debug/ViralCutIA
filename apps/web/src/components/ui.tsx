"use client";

import { Check, Copy } from "lucide-react";
import { useId, useState } from "react";

import { type ButtonSize, type ButtonVariant, buttonClass, cx } from "@/lib/classes";

export { buttonClass, cx };

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & { variant?: ButtonVariant; size?: ButtonSize };

export function Button({ variant = "secondary", size = "md", className, type = "button", ...rest }: ButtonProps) {
  return <button type={type} className={cx(buttonClass(variant, size), className)} {...rest} />;
}

interface SegmentedProps<T extends string | number> {
  label: string;
  value: T;
  options: { value: T; label: string; hint?: string }[];
  onChange: (value: T) => void;
  hideLabel?: boolean;
  disabled?: boolean;
}

/** A radio group drawn as a segmented control. */
export function Segmented<T extends string | number>({ label, value, options, onChange, hideLabel, disabled }: SegmentedProps<T>) {
  const name = useId();
  return (
    <fieldset disabled={disabled} className="min-w-0 disabled:opacity-50">
      <legend className={cx("mb-2 text-sm text-muted", hideLabel && "sr-only")}>{label}</legend>
      <div className="inline-flex max-w-full flex-wrap gap-1 rounded-[10px] border border-line bg-panel p-1">
        {options.map((o) => {
          const selected = o.value === value;
          return (
            <label
              key={String(o.value)}
              title={o.hint}
              className={cx(
                "relative rounded-md px-3 py-1.5 text-sm transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-amber",
                selected ? "bg-hover text-fg shadow-[inset_0_0_0_1px_var(--color-line-strong)]" : "text-muted hover:text-fg",
              )}
            >
              <input
                type="radio"
                name={name}
                className="sr-only"
                checked={selected}
                onChange={() => onChange(o.value)}
              />
              {o.label}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

export function Toggle({ label, hint, checked, onChange }: { label: string; hint?: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex items-start justify-between gap-6 py-1">
      <span>
        <span className="block text-sm">{label}</span>
        {hint && <span className="block text-[13px] text-faint">{hint}</span>}
      </span>
      <span className="relative mt-0.5 inline-flex shrink-0">
        <input type="checkbox" role="switch" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="h-6 w-10 rounded-full bg-line-strong transition-colors peer-checked:bg-amber peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-amber" />
        <span className="absolute left-0.5 top-0.5 size-5 rounded-full bg-fg transition-transform peer-checked:translate-x-4 peer-checked:bg-black" />
      </span>
    </label>
  );
}

export function Select<T extends string>({ label, value, options, onChange }: { label: string; value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  const id = useId();
  return (
    <div>
      <label htmlFor={id} className="mb-2 block text-sm text-muted">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
        className="h-10 w-full rounded-lg border border-line bg-panel px-3 text-sm text-fg"
      >
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  );
}

export function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      type="button"
      aria-label={copied ? "Copied" : label}
      title={copied ? "Copied" : label}
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          setCopied(true);
          setTimeout(() => setCopied(false), 1600);
        } catch {
          /* clipboard blocked: nothing to confirm */
        }
      }}
      className="inline-flex size-7 shrink-0 items-center justify-center rounded-md text-faint transition-colors hover:bg-hover hover:text-fg"
    >
      {copied ? <Check className="size-3.5 text-ok" aria-hidden /> : <Copy className="size-3.5" aria-hidden />}
    </button>
  );
}

export function ProgressBar({ value, label, indeterminate }: { value: number; label: string; indeterminate?: boolean }) {
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);
  return (
    <div
      role="progressbar"
      aria-label={label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={indeterminate ? undefined : pct}
      className="relative h-1 w-full overflow-hidden rounded-full bg-line"
    >
      {indeterminate ? (
        <div className="absolute inset-y-0 w-1/4 animate-sweep rounded-full bg-amber/70" />
      ) : (
        <div className="h-full rounded-full bg-amber transition-[width] duration-500 ease-out" style={{ width: `${pct}%` }} />
      )}
    </div>
  );
}

export function Notice({ tone = "info", children }: { tone?: "info" | "warn" | "error"; children: React.ReactNode }) {
  const tones = {
    info: "border-line text-muted",
    warn: "border-amber/30 text-fg",
    error: "border-danger/40 text-fg",
  };
  return (
    <div role={tone === "error" ? "alert" : "status"} className={cx("rounded-lg border bg-panel px-4 py-3 text-sm", tones[tone])}>
      {children}
    </div>
  );
}

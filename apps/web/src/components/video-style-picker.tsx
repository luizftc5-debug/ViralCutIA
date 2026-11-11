"use client";

import { useId } from "react";

import type { AppConfig, VideoStyle } from "@/lib/api";
import { VIDEO_STYLE_LABELS } from "@/lib/caption-styles";

import { cx } from "./ui";

const FALLBACK: AppConfig["video_styles"] = Object.entries(VIDEO_STYLE_LABELS)
  .filter(([k]) => k !== "dynamic" && k !== "editorial")
  .map(([key, label]) => ({ key: key as VideoStyle, label, description: "", caption: "auto", framing: "auto", best_for: [] }));

/** The video style library. Each style changes real editing parameters (see pipeline/styles.py). */
export function VideoStylePicker({ value, onChange, styles }: { value: VideoStyle; onChange: (v: VideoStyle) => void; styles?: AppConfig["video_styles"] }) {
  const name = useId();
  return (
    <fieldset>
      <legend className="mb-2 text-sm text-muted">Video style</legend>
      <div className="grid gap-2 sm:grid-cols-2">
        {(styles ?? FALLBACK).map((s) => {
          const selected = s.key === value;
          return (
            <label
              key={s.key}
              className={cx(
                "block rounded-xl border bg-panel px-3 py-2.5 transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-amber",
                selected ? "border-amber" : "border-line hover:border-line-strong",
              )}
            >
              <input type="radio" name={name} className="sr-only" checked={selected} onChange={() => onChange(s.key)} />
              <span className="block text-sm font-medium">{s.label}</span>
              {s.description && <span className="mt-0.5 block text-[13px] leading-snug text-faint">{s.description}</span>}
              {s.best_for.length > 0 && <span className="mt-1 block text-[12px] text-faint">Best for: {s.best_for.join(", ")}</span>}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

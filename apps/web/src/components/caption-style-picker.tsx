"use client";

import Image from "next/image";
import { useId } from "react";

import type { CaptionStyle } from "@/lib/api";
import { CAPTION_STYLES } from "@/lib/caption-styles";

import { cx } from "./ui";

/** Previews are frames rendered by the caption engine itself (scripts/render_style_previews.py). */
export function CaptionStylePicker({ value, onChange }: { value: CaptionStyle; onChange: (v: CaptionStyle) => void }) {
  const name = useId();
  return (
    <fieldset>
      <legend className="mb-2 text-sm text-muted">Caption style</legend>
      <label className={cx("mb-3 flex items-center gap-3 rounded-xl border bg-panel px-3 py-2.5 text-sm", value === "auto" ? "border-amber" : "border-line hover:border-line-strong")}>
        <input type="radio" name={name} className="sr-only" checked={value === "auto"} onChange={() => onChange("auto")} />
        <span className="font-medium">Match the video style</span>
        <span className="text-[13px] text-faint">Each clip uses the caption design of its video style.</span>
      </label>
      <div className="grid grid-cols-2 gap-3">
        {CAPTION_STYLES.map((s) => {
          const selected = s.key === value;
          return (
            <label
              key={s.key}
              className={cx(
                "block overflow-hidden rounded-xl border bg-panel transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-amber",
                selected ? "border-amber" : "border-line hover:border-line-strong",
              )}
            >
              <input type="radio" name={name} className="sr-only" checked={selected} onChange={() => onChange(s.key)} />
              <Image src={`/caption-styles/${s.key}.webp`} alt="" width={720} height={374} loading="lazy" className="aspect-[720/374] w-full object-cover" />
              <span className="block px-3 py-2.5">
                <span className="block text-sm font-medium">{s.label}</span>
                <span className="mt-0.5 block text-[13px] leading-snug text-faint">{s.description}</span>
              </span>
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}

"use client";

import type { AppConfig, CaptionOptions } from "@/lib/api";

import { CAPTION_STYLES } from "@/lib/caption-styles";

import { CaptionStylePicker } from "./caption-style-picker";
import { cx, Segmented, Select, Toggle } from "./ui";

const SWATCHES = ["#FFD60A", "#FFFFFF", "#3DFF8F", "#2F6BFF", "#FF5A5F", "#F2C879"];
const DEFAULT_FONTS: AppConfig["caption_fonts"] = [
  { key: "auto", label: "Style default" },
  { key: "inter", label: "Inter" },
  { key: "poppins", label: "Poppins" },
  { key: "anton", label: "Anton" },
  { key: "dm_serif", label: "DM Serif Display" },
];

/** Style, position, size, colour, typeface and animation. Shared by the settings form and the clip studio. */
export function CaptionControls({
  value,
  onChange,
  fonts,
  defaultAccent,
}: {
  value: CaptionOptions;
  onChange: (next: CaptionOptions) => void;
  fonts?: AppConfig["caption_fonts"];
  defaultAccent?: string;
}) {
  const set = <K extends keyof CaptionOptions>(key: K, v: CaptionOptions[K]) => onChange({ ...value, [key]: v });
  const styleLabel = CAPTION_STYLES.find((s) => s.key === value.style)?.label ?? "";
  return (
    <div className="space-y-6">
      <CaptionStylePicker value={value.style} onChange={(v) => set("style", v)} />
      <div className="grid gap-6 sm:grid-cols-2">
        <Segmented
          label="Position"
          value={value.position}
          onChange={(v) => set("position", v)}
          options={[
            { value: "top", label: "Top" },
            { value: "middle", label: "Middle" },
            { value: "bottom", label: "Bottom" },
          ]}
        />
        <Segmented
          label="Size"
          value={value.size}
          onChange={(v) => set("size", v)}
          options={[
            { value: "s", label: "Small" },
            { value: "m", label: "Medium" },
            { value: "l", label: "Large" },
          ]}
        />
      </div>
      <div className="grid gap-6 sm:grid-cols-2">
        <fieldset>
          <legend className="mb-2 text-sm text-muted">Accent colour</legend>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              onClick={() => set("color", "")}
              aria-pressed={value.color === ""}
              className={cx("h-8 rounded-md border px-2.5 text-[13px]", value.color === "" ? "border-amber text-fg" : "border-line text-muted hover:text-fg")}
            >
              {styleLabel ? "Style default" : "Default"}
            </button>
            {SWATCHES.map((c) => (
              <button
                key={c}
                type="button"
                onClick={() => set("color", c)}
                aria-pressed={value.color === c}
                aria-label={`Accent ${c}`}
                className={cx("size-8 rounded-md border-2", value.color === c ? "border-fg" : "border-transparent")}
                style={{ backgroundColor: c }}
              />
            ))}
            <label className="relative inline-flex h-8 items-center rounded-md border border-line px-2.5 text-[13px] text-muted hover:text-fg">
              Custom
              <input
                type="color"
                value={value.color || defaultAccent || "#FFD60A"}
                onChange={(e) => set("color", e.target.value.toUpperCase())}
                className="absolute inset-0 size-full opacity-0"
                aria-label="Custom accent colour"
              />
            </label>
          </div>
        </fieldset>
        <Select label="Typeface" value={value.font} onChange={(v) => set("font", v)} options={(fonts ?? DEFAULT_FONTS).map((f) => ({ value: f.key, label: f.label }))} />
      </div>
      <div className="divide-y divide-line border-y border-line">
        <Toggle label="Highlight key words" hint="Recurring topic words and numbers take the accent colour." checked={value.highlight_keywords} onChange={(v) => set("highlight_keywords", v)} />
        <Toggle label="Animate words" hint="A short pop or reveal as each word is spoken." checked={value.animation} onChange={(v) => set("animation", v)} />
      </div>
    </div>
  );
}

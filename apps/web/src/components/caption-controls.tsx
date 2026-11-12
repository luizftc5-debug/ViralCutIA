"use client";

import type { AppConfig, CaptionOptions, JobOptions } from "@/lib/api";

import { CAPTION_STYLES } from "@/lib/caption-styles";

import { CaptionPreview } from "./caption-preview";
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
  preview,
}: {
  value: CaptionOptions;
  onChange: (next: CaptionOptions) => void;
  fonts?: AppConfig["caption_fonts"];
  defaultAccent?: string;
  /** Show the live preview (rendered by the export engine); optional job/clip for a real background frame. */
  preview?: { platform?: JobOptions["platform"]; jobId?: string; clipId?: string; language?: string; autoStyle?: CaptionOptions["style"] };
}) {
  const set = <K extends keyof CaptionOptions>(key: K, v: CaptionOptions[K]) => onChange({ ...value, [key]: v });
  const styleLabel = CAPTION_STYLES.find((s) => s.key === value.style)?.label ?? "";
  const previewStyle = value.style === "auto" ? (preview?.autoStyle ?? "high_retention") : value.style;
  return (
    <div className="space-y-6">
      <div className={preview ? "grid gap-6 md:grid-cols-[minmax(0,1fr)_240px]" : ""}>
        <CaptionStylePicker value={value.style} onChange={(v) => set("style", v)} />
        {preview && (
          <div className="md:sticky md:top-24 md:self-start">
            <CaptionPreview captions={{ ...value, style: previewStyle }} platform={preview.platform} jobId={preview.jobId} clipId={preview.clipId} language={preview.language} />
          </div>
        )}
      </div>
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
            { value: "s", label: "S" },
            { value: "m", label: "M" },
            { value: "l", label: "L" },
            { value: "xl", label: "XL" },
          ]}
        />
      </div>
      <div className="grid gap-6 sm:grid-cols-2">
        <fieldset>
          <legend className="mb-2 text-sm text-muted">Highlight colour</legend>
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
      <CaptionStudio value={value} onChange={onChange} />
    </div>
  );
}

function ColorField({ label, value, onChange }: { label: string; value: string; onChange: (v: string) => void }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <span className="w-28">{label}</span>
      <button type="button" onClick={() => onChange("")} aria-pressed={!value} className={cx("h-8 rounded-md border px-2.5 text-[13px]", !value ? "border-amber text-fg" : "border-line hover:text-fg")}>
        Style
      </button>
      <label className="relative inline-flex h-8 items-center gap-2 rounded-md border border-line px-2.5 text-[13px] hover:text-fg">
        <span className="size-4 rounded-sm border border-line" style={{ backgroundColor: value || "transparent" }} />
        Custom
        <input type="color" value={value || "#FFFFFF"} onChange={(e) => onChange(e.target.value.toUpperCase())} className="absolute inset-0 size-full opacity-0" aria-label={`${label} colour`} />
      </label>
    </div>
  );
}

function Slider({ label, value, min, max, step, auto, onChange, format }: { label: string; value: number | null | undefined; min: number; max: number; step: number; auto: string; onChange: (v: number | null) => void; format?: (v: number) => string }) {
  const on = value !== null && value !== undefined;
  return (
    <label className="block">
      <span className="mb-1 flex justify-between text-sm text-muted">
        {label}
        <span className="tnum text-fg">
          {on ? (format ? format(value) : value) : auto}
          {on && (
            <button type="button" className="ml-2 text-[12px] text-faint underline" onClick={(e) => { e.preventDefault(); onChange(null); }}>
              reset
            </button>
          )}
        </span>
      </span>
      <input type="range" min={min} max={max} step={step} value={on ? value : min} onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
    </label>
  );
}

/** Fine control over the caption design. Every field changes the burned-in captions of the export. */
function CaptionStudio({ value, onChange }: { value: CaptionOptions; onChange: (next: CaptionOptions) => void }) {
  const set = <K extends keyof CaptionOptions>(key: K, v: CaptionOptions[K]) => onChange({ ...value, [key]: v });
  return (
    <details className="rounded-xl border border-line bg-panel/40 p-4">
      <summary className="text-sm font-medium">Caption Studio: fine-tune the design</summary>
      <div className="mt-5 grid gap-6 sm:grid-cols-2">
        <Select label="Weight" value={value.weight ?? "auto"} onChange={(v) => set("weight", v)} options={[{ value: "auto", label: "Style default" }, { value: "medium", label: "Medium" }, { value: "bold", label: "Bold" }, { value: "heavy", label: "Heavy" }]} />
        <Select label="Capitalisation" value={value.case ?? "auto"} onChange={(v) => set("case", v)} options={[{ value: "auto", label: "Style default" }, { value: "upper", label: "ALL CAPS" }, { value: "normal", label: "Standard" }]} />
        <Select label="Animation" value={value.animation_type ?? "auto"} onChange={(v) => set("animation_type", v)} options={[{ value: "auto", label: "Style default" }, { value: "color", label: "Colour highlight" }, { value: "pop", label: "Pop" }, { value: "box", label: "Highlight box" }, { value: "reveal", label: "Reveal" }, { value: "none", label: "None" }]} />
        <Segmented label="Animation intensity" value={value.animation_intensity ?? "normal"} onChange={(v) => set("animation_intensity", v)} options={[{ value: "subtle", label: "Subtle" }, { value: "normal", label: "Normal" }, { value: "strong", label: "Strong" }]} />
        <Select label="Keywords" value={value.keyword_mode ?? "auto"} onChange={(v) => set("keyword_mode", v)} options={[{ value: "auto", label: "Style default" }, { value: "color", label: "Colour" }, { value: "color_scale", label: "Colour + larger" }, { value: "off", label: "No keyword emphasis" }]} />
        <Select label="Background" value={value.background ?? "auto"} onChange={(v) => set("background", v)} options={[{ value: "auto", label: "Style default" }, { value: "none", label: "None" }, { value: "box", label: "Soft panel" }]} />
        <ColorField label="Text colour" value={value.text_color ?? ""} onChange={(v) => set("text_color", v)} />
        <ColorField label="Keyword colour" value={value.keyword_color ?? ""} onChange={(v) => set("keyword_color", v)} />
        <Slider label="Words on screen" value={value.words_per_page} min={1} max={8} step={1} auto="Style" onChange={(v) => set("words_per_page", v)} />
        <Slider label="Lines" value={value.max_lines} min={1} max={3} step={1} auto="Style" onChange={(v) => set("max_lines", v)} />
        <Slider label="Line spacing" value={value.line_spacing} min={0.8} max={1.6} step={0.02} auto="Style" onChange={(v) => set("line_spacing", v)} format={(v) => v.toFixed(2)} />
        <Slider label="Outline" value={value.outline} min={0} max={16} step={0.5} auto="Style" onChange={(v) => set("outline", v)} />
        <Slider label="Shadow" value={value.shadow} min={0} max={12} step={0.5} auto="Style" onChange={(v) => set("shadow", v)} />
        <Slider label="Vertical nudge" value={value.offset || null} min={-0.2} max={0.2} step={0.01} auto="0" onChange={(v) => set("offset", v ?? 0)} format={(v) => `${Math.round(v * 100)}%`} />
      </div>
      <p className="mt-4 text-[12px] text-faint">Captions always stay inside the platform’s safe area; a page that would overflow is scaled down, never clipped.</p>
    </details>
  );
}

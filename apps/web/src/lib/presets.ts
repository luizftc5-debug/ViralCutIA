import { DEFAULT_OPTIONS, type EditorsConfig, type JobOptions, type PresetInfo } from "./api";

/** Mirrors pipeline/editors.py apply_preset: a preset sets real options and switches editors on. */
export function applyPreset(options: JobOptions, preset: PresetInfo): JobOptions {
  const next: JobOptions = { ...options, preset: preset.key };
  for (const [k, v] of Object.entries(preset.options)) {
    if (k === "clip_count" && options.clip_count !== "auto") continue; // the user's own number wins
    if (k === "captions") next.captions = { ...options.captions, ...(v as object) };
    else (next as unknown as Record<string, unknown>)[k] = v;
  }
  const editors: EditorsConfig = {};
  for (const [k, cfg] of Object.entries(options.editors)) editors[k] = { ...cfg };
  for (const [k, cfg] of Object.entries(preset.editors)) editors[k] = { ...(editors[k] ?? {}), ...cfg };
  next.editors = editors;
  return next;
}

/**
 * Pick `preset` in place of `previous`. What the previous preset had set goes back to the default
 * first, so choosing "Podcast" after "Faceless" does not keep Faceless' blurred framing and editors.
 */
export function switchPreset(options: JobOptions, preset: PresetInfo, previous?: PresetInfo): JobOptions {
  if (!previous || previous.key === preset.key) return applyPreset(options, preset);
  const base: JobOptions = { ...options, editors: { ...options.editors } };
  const defaults = DEFAULT_OPTIONS as unknown as Record<string, unknown>;
  for (const [k, v] of Object.entries(previous.options)) {
    if (k === "clip_count") continue;
    if (k === "captions") {
      const captions = { ...base.captions } as unknown as Record<string, unknown>;
      for (const ck of Object.keys(v as object)) captions[ck] = (DEFAULT_OPTIONS.captions as unknown as Record<string, unknown>)[ck];
      base.captions = captions as unknown as JobOptions["captions"];
    } else (base as unknown as Record<string, unknown>)[k] = defaults[k];
  }
  for (const k of Object.keys(previous.editors)) delete base.editors[k];
  return applyPreset(base, preset);
}

const CONTENT_LABEL: Record<JobOptions["content_type"], string> = {
  shorts: "YouTube Shorts",
  tiktok: "TikTok",
  reels: "Instagram Reels",
  youtube: "YouTube long-form",
  compilation: "Compilation",
};

/** One line that says what "Generate" will make, shown next to the button. */
export function summarize(options: JobOptions, presets: PresetInfo[] | undefined): string {
  const preset = presets?.find((p) => p.key === (options.preset || "auto"));
  const wide = options.content_type === "youtube" || (options.content_type === "compilation" && options.aspect === "16:9");
  return [
    `${CONTENT_LABEL[options.content_type]} (${wide ? "16:9" : "9:16"})`,
    preset ? preset.label : "Automatic style",
    options.captions.enabled ? "captions on" : "no captions",
    options.clip_count === "auto" ? "" : `up to ${options.clip_count} clip${options.clip_count === 1 ? "" : "s"}`,
  ]
    .filter(Boolean)
    .join(" · ");
}

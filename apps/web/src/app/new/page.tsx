"use client";

import { FileVideo, RotateCcw, X } from "lucide-react";
import dynamic from "next/dynamic";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { CaptionControls } from "@/components/caption-controls";
import { SiteHeader } from "@/components/site-header";
import { TimelineDropzone } from "@/components/timeline-dropzone";
import { applyPreset, ContentTypePicker, PresetGrid } from "@/components/creator-studio";
import { VideoStylePicker } from "@/components/video-style-picker";

// the 13-editor panel is only downloaded when Advanced mode is opened
const EditorsPanel = dynamic(() => import("@/components/creator-studio").then((m) => m.EditorsPanel), {
  loading: () => <p className="text-sm text-faint">Loading editors…</p>,
});
import { Button, cx, Notice, ProgressBar, Segmented, Select, Toggle } from "@/components/ui";
import { useUpload } from "@/components/upload-provider";
import { api, DEFAULT_OPTIONS, type JobOptions } from "@/lib/api";
import { fileSize, LANGUAGE_OPTIONS, timecode } from "@/lib/format";

const INTENSITY_NOTES: Record<JobOptions["intensity"], string> = {
  subtle: "Keeps more of the original rhythm: longer pauses, fewer and smaller moves.",
  balanced: "The style as designed.",
  high: "Tighter cuts and more frequent punch-ins, within limits that keep it watchable.",
};

export default function NewVideoPage() {
  const upload = useUpload();
  const router = useRouter();
  const [options, setOptions] = useState<JobOptions>(DEFAULT_OPTIONS);
  const [customCount, setCustomCount] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");
  const [mode, setMode] = useState<"basic" | "advanced">("basic");

  const set = <K extends keyof JobOptions>(key: K, v: JobOptions[K]) => setOptions((o) => ({ ...o, [key]: v }));
  const minScoreDefault = upload.config?.engine.min_score ?? 60;

  const generate = async (override?: JobOptions) => {
    if (!upload.job) return;
    setStarting(true);
    setStartError("");
    try {
      const jobId = upload.job.id;
      await api.start(jobId, override ?? options);
      upload.reset();
      router.push(`/jobs/${jobId}`);
    } catch (e) {
      setStartError(e instanceof Error ? e.message : "Could not start processing.");
      setStarting(false);
    }
  };

  if (upload.phase === "idle") {
    return (
      <>
        <SiteHeader studio />
        <main id="main" className="mx-auto max-w-[1240px] px-5 py-14 sm:px-8">
          <h1 className="text-4xl font-semibold">Start with a video</h1>
          <p className="mt-3 max-w-xl text-muted">Pick the long video you want clips from. You choose captions and clip settings while it uploads.</p>
          <div className="mt-8">
            <TimelineDropzone compact />
          </div>
        </main>
      </>
    );
  }

  const media = upload.job?.media;
  const pickedCount = options.clip_count;
  const countChoice = customCount ? "custom" : pickedCount === "auto" ? "auto" : [3, 5, 10].includes(pickedCount) ? String(pickedCount) : "custom";

  return (
    <>
      <SiteHeader studio />
      <main id="main" className="mx-auto max-w-[1240px] px-5 pb-32 pt-10 sm:px-8">
        <div className="grid gap-x-14 gap-y-10 lg:grid-cols-[minmax(0,4fr)_minmax(0,7fr)]">
          {/* source file */}
          <aside className="lg:sticky lg:top-24 lg:self-start">
            <h1 className="text-3xl font-semibold">Your video</h1>
            <div className="mt-5 rounded-(--radius-frame) border border-line bg-panel p-5">
              <div className="flex items-start gap-3">
                <FileVideo className="mt-0.5 size-5 shrink-0 text-muted" aria-hidden />
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium" title={upload.file?.name}>
                    {upload.file?.name}
                  </p>
                  <p className="tnum text-sm text-faint">
                    {fileSize(upload.file?.size)}
                    {media?.duration ? `, ${timecode(media.duration)} long, ${media.width} × ${media.height}` : ""}
                  </p>
                </div>
                <button type="button" onClick={upload.reset} aria-label="Remove this video" title="Remove this video" className="rounded-md p-1 text-faint hover:bg-hover hover:text-fg">
                  <X className="size-4" aria-hidden />
                </button>
              </div>

              <div className="mt-5" aria-live="polite">
                {upload.phase === "uploading" && (
                  <>
                    <ProgressBar value={upload.progress} label="Upload progress" />
                    <p className="tnum mt-2 text-sm text-muted">
                      {upload.progress < 1 ? `Uploading, ${Math.round(upload.progress * 100)}%` : "Checking the file…"}
                    </p>
                  </>
                )}
                {upload.phase === "ready" && <p className="text-sm text-ok">Uploaded and checked. Ready to process.</p>}
                {upload.phase === "error" && (
                  <div className="space-y-3">
                    <Notice tone="error">{upload.error}</Notice>
                    <Button size="sm" onClick={() => upload.file && upload.begin(upload.file)}>
                      <RotateCcw className="size-3.5" aria-hidden />
                      Try the upload again
                    </Button>
                  </div>
                )}
              </div>
            </div>
            {!media?.has_audio && upload.phase === "ready" && (
              <div className="mt-4">
                <Notice tone="warn">This video has no audio track. Clips will be chosen from the picture alone and can’t have captions.</Notice>
              </div>
            )}
          </aside>

          {/* settings */}
          <form
            className="space-y-12"
            onSubmit={(e) => {
              e.preventDefault();
              void generate();
            }}
          >
            <section aria-labelledby="studio-h" className="space-y-8">
              <div className="flex flex-wrap items-end justify-between gap-4">
                <div>
                  <h2 id="studio-h" className="text-2xl font-semibold">
                    Creator Studio
                  </h2>
                  <p className="mt-1 text-sm text-muted">One click, a preset, or every editor by hand.</p>
                </div>
                <Segmented
                  label="Interface"
                  hideLabel
                  value={mode}
                  onChange={(v) => setMode(v)}
                  options={[
                    { value: "basic", label: "Basic" },
                    { value: "advanced", label: "Advanced" },
                  ]}
                />
              </div>
              <ContentTypePicker
                value={options.content_type}
                onChange={(v) => setOptions((o) => ({ ...o, content_type: v, aspect: v === "youtube" ? "16:9" : o.aspect, clip_count: v === "compilation" && o.clip_count === "auto" ? 5 : o.clip_count }))}
              />
              {options.content_type === "compilation" && (
                <Segmented
                  label="Compilation format"
                  value={options.aspect}
                  onChange={(v) => set("aspect", v)}
                  options={[
                    { value: "9:16", label: "Vertical 9:16" },
                    { value: "16:9", label: "Horizontal 16:9" },
                  ]}
                />
              )}

              <div className="rounded-xl border border-line bg-panel p-5">
                <h3 className="text-lg font-semibold">A. Quick generation</h3>
                <p className="mt-1 text-sm text-muted">The AI analyses the video, chooses the editing strategy per clip and renders publication-ready files.</p>
                <Button
                  variant="primary"
                  className="mt-4"
                  disabled={upload.phase !== "ready" || starting}
                  onClick={() => {
                    const auto = upload.config?.presets?.find((p) => p.key === "auto");
                    const next = auto ? applyPreset(options, auto) : options;
                    setOptions(next);
                    void generate(next);
                  }}
                >
                  Generate automatically
                </Button>
              </div>

              <div>
                <h3 className="text-lg font-semibold">B. Creative presets</h3>
                <p className="mt-1 mb-3 text-sm text-muted">Each preset sets real pacing, captions, framing and editors. You can still change anything after picking one.</p>
                {upload.config?.presets ? <PresetGrid presets={upload.config.presets} value={options.preset} onPick={(p) => setOptions((o) => applyPreset(o, p))} /> : <p className="text-sm text-faint">Loading presets…</p>}
              </div>

              {mode === "advanced" && (
                <div>
                  <h3 className="text-lg font-semibold">C. Advanced editors</h3>
                  <p className="mt-1 mb-3 text-sm text-muted">Thirteen specialised editors. Each switch changes the exported video.</p>
                  <EditorsPanel
                    editors={options.editors}
                    onChange={(e) => set("editors", e)}
                    info={upload.config?.editors}
                    jobId={upload.phase === "ready" ? upload.job?.id : undefined}
                    captionsOn={options.captions.enabled}
                    onCaptions={(v) => set("captions", { ...options.captions, enabled: v })}
                  />
                </div>
              )}
            </section>

            <section aria-labelledby="captions-q" className="border-t border-line pt-10">
              <h2 id="captions-q" className="text-2xl font-semibold">
                Do you want captions on your clips?
              </h2>
              <div className="mt-5 grid gap-3 sm:grid-cols-2" role="radiogroup" aria-labelledby="captions-q">
                {[
                  { on: true, title: "Yes", body: "Automatic captions, synced to each spoken word." },
                  { on: false, title: "No", body: "Clean cuts with nothing drawn on the picture." },
                ].map((c) => {
                  const selected = options.captions.enabled === c.on;
                  return (
                    <label
                      key={c.title}
                      className={cx(
                        "block rounded-xl border bg-panel p-4 transition-colors has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-amber",
                        selected ? "border-amber" : "border-line hover:border-line-strong",
                      )}
                    >
                      <input type="radio" name="captions-enabled" className="sr-only" checked={selected} onChange={() => set("captions", { ...options.captions, enabled: c.on })} />
                      <span className="block font-display text-xl font-semibold">{c.title}</span>
                      <span className="mt-1 block text-sm text-muted">{c.body}</span>
                    </label>
                  );
                })}
              </div>

              {options.captions.enabled && mode === "basic" ? (
                <p className="mt-4 text-sm text-faint">Caption design follows the preset. Switch to Advanced for every caption style and Caption Studio.</p>
              ) : options.captions.enabled ? (
                <div className="mt-8 space-y-6">
                  <CaptionControls
                    value={options.captions}
                    onChange={(c) => set("captions", c)}
                    fonts={upload.config?.caption_fonts}
                    defaultAccent={upload.config?.caption_styles.find((s) => s.key === options.captions.style)?.accent}
                    preview={{
                      platform: options.platform,
                      jobId: upload.phase === "ready" ? upload.job?.id : undefined,
                      language: options.language === "auto" ? undefined : options.language,
                      autoStyle: upload.config?.video_styles?.find((v) => v.key === options.style)?.caption,
                    }}
                  />
                </div>
              ) : (
                <p className="mt-4 text-sm text-faint">You still get an SRT subtitle file with each clip, in case you want captions added by the platform.</p>
              )}
              <div className="mt-6 max-w-xs">
                <Select label="Spoken language" value={options.language} onChange={(v) => set("language", v)} options={LANGUAGE_OPTIONS} />
              </div>
            </section>

            {mode === "advanced" && (
            <section aria-labelledby="clips-h" className="space-y-6 border-t border-line pt-10">
              <h2 id="clips-h" className="text-2xl font-semibold">
                Clips
              </h2>
              <div>
                <Segmented
                  label="How many"
                  value={countChoice}
                  onChange={(v) => {
                    if (v === "custom") {
                      setCustomCount(true);
                      if (pickedCount === "auto") set("clip_count", 4);
                    } else {
                      setCustomCount(false);
                      set("clip_count", v === "auto" ? "auto" : Number(v));
                    }
                  }}
                  options={[
                    { value: "auto", label: "Automatic" },
                    { value: "3", label: "3" },
                    { value: "5", label: "5" },
                    { value: "10", label: "10" },
                    { value: "custom", label: "Custom" },
                  ]}
                />
                {countChoice === "custom" && (
                  <label className="mt-3 flex items-center gap-3 text-sm text-muted">
                    Number of clips
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={pickedCount === "auto" ? 4 : pickedCount}
                      onChange={(e) => set("clip_count", Math.max(1, Math.min(20, Number(e.target.value) || 1)))}
                      className="tnum h-10 w-20 rounded-lg border border-line bg-panel px-3 text-fg"
                    />
                  </label>
                )}
                <p className="mt-2 text-[13px] text-faint">This is a ceiling. You get fewer clips when fewer moments reach the minimum score.</p>
              </div>
              <div>
                <Segmented
                  label="Length"
                  value={options.duration}
                  onChange={(v) => set("duration", v)}
                  options={[
                    { value: "auto", label: "Automatic" },
                    { value: "15-30", label: "15–30 s" },
                    { value: "30-60", label: "30–60 s" },
                    { value: "60-90", label: "60–90 s" },
                    { value: "custom", label: "Custom" },
                  ]}
                />
                {options.duration === "custom" && (
                  <div className="mt-3 flex items-center gap-3 text-sm text-muted">
                    From
                    <input type="number" min={5} max={175} value={options.custom_min} onChange={(e) => set("custom_min", Math.max(5, Math.min(175, Number(e.target.value) || 5)))} className="tnum h-10 w-20 rounded-lg border border-line bg-panel px-3 text-fg" aria-label="Minimum length in seconds" />
                    to
                    <input type="number" min={10} max={180} value={options.custom_max} onChange={(e) => set("custom_max", Math.max(10, Math.min(180, Number(e.target.value) || 10)))} className="tnum h-10 w-20 rounded-lg border border-line bg-panel px-3 text-fg" aria-label="Maximum length in seconds" />
                    seconds
                  </div>
                )}
              </div>
              <Segmented
                label="Platform"
                value={options.platform}
                onChange={(v) => set("platform", v)}
                options={[
                  { value: "universal", label: "Universal" },
                  { value: "shorts", label: "Shorts" },
                  { value: "reels", label: "Reels" },
                  { value: "tiktok", label: "TikTok" },
                ]}
              />
              <p className="-mt-3 text-[13px] text-faint">Sets how far captions stay from each app’s buttons and text. Universal clears all three.</p>
              <VideoStylePicker
                value={options.style}
                styles={upload.config?.video_styles}
                onChange={(v) => {
                  set("style", v);
                  // a video style comes with its caption design; the user can still change it below
                  if (options.captions.enabled && options.captions.style !== "auto") {
                    const cap = upload.config?.video_styles?.find((x) => x.key === v)?.caption;
                    if (cap && cap !== "auto") set("captions", { ...options.captions, style: cap });
                  }
                }}
              />
              <div>
                <Segmented
                  label="Editing intensity"
                  value={options.intensity}
                  onChange={(v) => set("intensity", v)}
                  options={[
                    { value: "subtle", label: "Subtle" },
                    { value: "balanced", label: "Balanced" },
                    { value: "high", label: "High energy" },
                  ]}
                />
                <p className="mt-2 text-[13px] text-faint">{INTENSITY_NOTES[options.intensity]}</p>
              </div>
              <Segmented
                label="Visual format"
                value={options.framing}
                onChange={(v) => set("framing", v)}
                options={[
                  { value: "auto", label: "Smart auto" },
                  { value: "face", label: "Full-screen vertical" },
                  { value: "speaker", label: "Speaker focus" },
                  { value: "split", label: "Split-screen" },
                  { value: "blur", label: "Blurred background" },
                ]}
              />
              <div>
                <Segmented
                  label="Versions per clip"
                  value={String(options.variations)}
                  onChange={(v) => set("variations", Number(v))}
                  options={[
                    { value: "1", label: "1" },
                    { value: "2", label: "2" },
                    { value: "3", label: "3" },
                  ]}
                />
                <p className="mt-2 text-[13px] text-faint">Extra versions reuse the same analysis: a hook-first cut when a stronger opening exists, another style, or another pacing.</p>
              </div>
              <Segmented
                label="Export quality"
                value={options.quality}
                onChange={(v) => set("quality", v)}
                options={[
                  { value: "high", label: "High quality" },
                  { value: "max", label: "Maximum quality" },
                ]}
              />
              <p className="text-sm text-muted">Format: {options.content_type === "youtube" || (options.content_type === "compilation" && options.aspect === "16:9") ? "horizontal 16:9" : "vertical 9:16"}.</p>
            </section>
            )}

            {mode === "advanced" && (
            <details className="border-t border-line pt-6">
              <summary className="text-sm text-muted hover:text-fg">Advanced</summary>
              <div className="mt-6 space-y-6">
                <label className="block max-w-md">
                  <span className="mb-2 flex justify-between text-sm text-muted">
                    Minimum score to keep a clip
                    <span className="tnum text-fg">{options.min_score ?? minScoreDefault}</span>
                  </span>
                  <input type="range" min={0} max={90} step={1} value={options.min_score ?? minScoreDefault} onChange={(e) => set("min_score", Number(e.target.value))} className="w-full" />
                  <span className="mt-1 block text-[13px] text-faint">Lower it to get more clips from a weak video; raise it to keep only the strongest.</span>
                </label>
                <Toggle
                  label="Review the transcript before rendering"
                  hint="Pauses after the clips are chosen so you can fix misheard words. Nothing is rendered until you continue."
                  checked={options.review_transcript}
                  onChange={(v) => set("review_transcript", v)}
                />
                <Toggle
                  label="Smart punch-in zoom"
                  hint="A small, eased zoom on words the speaker really stresses. Only on clips that follow a face."
                  checked={options.smart_zoom}
                  onChange={(v) => set("smart_zoom", v)}
                />
                <label className="block max-w-md">
                  <span className="mb-2 block text-sm text-muted">Names and terms (optional)</span>
                  <input
                    type="text"
                    value={options.glossary}
                    maxLength={400}
                    placeholder="e.g. ViralCut, Kubernetes, Dra. Almeida"
                    onChange={(e) => set("glossary", e.target.value)}
                    className="h-11 w-full rounded-lg border border-line bg-panel px-3 text-fg placeholder:text-faint"
                  />
                  <span className="mt-1 block text-[13px] text-faint">Comma separated. The speech recogniser prefers these spellings.</span>
                </label>
                <div>
                  <Segmented
                    label="Processing mode"
                    value={options.performance}
                    onChange={(v) => set("performance", v)}
                    options={[
                      { value: "auto", label: `Recommended${upload.config?.performance ? ` (${upload.config.performance.recommended})` : ""}` },
                      { value: "eco", label: "Eco" },
                      { value: "balanced", label: "Balanced" },
                      { value: "turbo", label: "Turbo" },
                    ]}
                  />
                  <p className="mt-2 text-[13px] text-faint">
                    {upload.config?.performance
                      ? `Processing server: ${upload.config.performance.hardware.logical_cpus} CPU threads, ${upload.config.performance.hardware.ram_gb ?? "?"} GB RAM${upload.config.performance.hardware.gpu?.name ? `, ${upload.config.performance.hardware.gpu.name}` : ""}. ${upload.config.performance.reason}. `
                      : ""}
                    Eco uses half the CPU at low priority; Turbo uses all of it. Output quality is the same in every mode.
                  </p>
                </div>
              </div>
            </details>
            )}

            {startError && <Notice tone="error">{startError}</Notice>}

            <div className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-black/90 backdrop-blur">
              <div className="mx-auto flex max-w-[1240px] items-center justify-between gap-4 px-5 py-4 sm:px-8">
                <p className="hidden text-sm text-muted sm:block" aria-live="polite">
                  {upload.phase === "uploading" && "You can generate as soon as the upload finishes."}
                  {upload.phase === "ready" && "Processing runs on the server. You can leave this page and come back."}
                  {upload.phase === "error" && "Fix the upload to continue."}
                </p>
                <Button type="submit" variant="primary" size="lg" disabled={upload.phase !== "ready" || starting} className="w-full sm:w-auto">
                  {starting ? "Starting…" : "Generate Viral Clips"}
                </Button>
              </div>
            </div>
          </form>
        </div>
      </main>
    </>
  );
}

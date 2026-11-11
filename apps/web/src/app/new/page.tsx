"use client";

import { FileVideo, RotateCcw, X } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { CaptionControls } from "@/components/caption-controls";
import { SiteHeader } from "@/components/site-header";
import { TimelineDropzone } from "@/components/timeline-dropzone";
import { Button, cx, Notice, ProgressBar, Segmented, Select } from "@/components/ui";
import { useUpload } from "@/components/upload-provider";
import { api, DEFAULT_OPTIONS, type JobOptions } from "@/lib/api";
import { fileSize, LANGUAGE_OPTIONS, timecode } from "@/lib/format";

const STYLE_NOTES: Record<JobOptions["style"], string> = {
  auto: "Balanced pacing: silent pauses longer than 0.7 s are shortened.",
  dynamic: "Fast pacing: silent pauses longer than 0.45 s are shortened.",
  editorial: "Unhurried: only pauses longer than a second are shortened.",
  podcast: "Conversation pacing, with split screen when two people can’t share the frame.",
  gaming: "No pause trimming, and the full frame is kept over a blurred backdrop so off-centre action stays visible.",
};

export default function NewVideoPage() {
  const upload = useUpload();
  const router = useRouter();
  const [options, setOptions] = useState<JobOptions>(DEFAULT_OPTIONS);
  const [customCount, setCustomCount] = useState(false);
  const [starting, setStarting] = useState(false);
  const [startError, setStartError] = useState("");

  const set = <K extends keyof JobOptions>(key: K, v: JobOptions[K]) => setOptions((o) => ({ ...o, [key]: v }));
  const minScoreDefault = upload.config?.engine.min_score ?? 60;

  const generate = async () => {
    if (!upload.job) return;
    setStarting(true);
    setStartError("");
    try {
      const jobId = upload.job.id;
      await api.start(jobId, options);
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
            <section aria-labelledby="captions-q">
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

              {options.captions.enabled ? (
                <div className="mt-8 space-y-6">
                  <CaptionControls
                    value={options.captions}
                    onChange={(c) => set("captions", c)}
                    fonts={upload.config?.caption_fonts}
                    defaultAccent={upload.config?.caption_styles.find((s) => s.key === options.captions.style)?.accent}
                  />
                </div>
              ) : (
                <p className="mt-4 text-sm text-faint">You still get an SRT subtitle file with each clip, in case you want captions added by the platform.</p>
              )}
              <div className="mt-6 max-w-xs">
                <Select label="Spoken language" value={options.language} onChange={(v) => set("language", v)} options={LANGUAGE_OPTIONS} />
              </div>
            </section>

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
              <Segmented
                label="Length"
                value={options.duration}
                onChange={(v) => set("duration", v)}
                options={[
                  { value: "auto", label: "Automatic" },
                  { value: "15-30", label: "15–30 s" },
                  { value: "30-60", label: "30–60 s" },
                  { value: "60-90", label: "60–90 s" },
                ]}
              />
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
              <div>
                <Segmented
                  label="Editing style"
                  value={options.style}
                  onChange={(v) => set("style", v)}
                  options={[
                    { value: "auto", label: "Automatic" },
                    { value: "dynamic", label: "Dynamic" },
                    { value: "editorial", label: "Editorial" },
                    { value: "podcast", label: "Podcast" },
                    { value: "gaming", label: "Gaming" },
                  ]}
                />
                <p className="mt-2 text-[13px] text-faint">{STYLE_NOTES[options.style]}</p>
              </div>
              <p className="text-sm text-muted">Format: vertical 9:16.</p>
            </section>

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
                <Segmented
                  label="Framing"
                  value={options.framing}
                  onChange={(v) => set("framing", v)}
                  options={[
                    { value: "auto", label: "Automatic" },
                    { value: "face", label: "Follow the speaker" },
                    { value: "split", label: "Split screen" },
                    { value: "blur", label: "Blurred backdrop" },
                    { value: "center", label: "Centre crop" },
                  ]}
                />
              </div>
            </details>

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

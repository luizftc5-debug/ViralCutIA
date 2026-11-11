"use client";

import { Captions, Download, PenLine } from "lucide-react";
import Link from "next/link";

import { type Clip, mediaUrl } from "@/lib/api";
import { languageName, layoutName, seconds, timecode } from "@/lib/format";

import { ScoreBreakdown, ScoreNumber } from "./score";
import { buttonClass, CopyButton, Notice, ProgressBar } from "./ui";

export function ClipRow({ clip, jobId }: { clip: Clip; jobId: string }) {
  const rendering = clip.status === "queued" || clip.status === "rendering" || clip.status === "pending";
  const hookLabels = (clip.hook.types ?? []).map((t) => t.label).join(", ");

  return (
    <li className="grid gap-x-8 gap-y-6 py-10 md:grid-cols-[240px_minmax(0,1fr)]" aria-label={`Clip ${clip.rank}`}>
      <div>
        <div className="relative aspect-[9/16] w-full max-w-[240px] overflow-hidden rounded-(--radius-frame) border border-line bg-panel">
          {clip.urls ? (
            <video
              key={clip.version}
              controls
              playsInline
              preload="none"
              poster={mediaUrl(clip.urls.thumbnail)}
              src={mediaUrl(clip.urls.video)}
              className="size-full object-cover"
              aria-label={`Preview of clip ${clip.rank}`}
            />
          ) : (
            <div className="flex size-full items-center justify-center p-6 text-center text-sm text-faint">{clip.status === "failed" ? "No file" : "Rendering…"}</div>
          )}
          {rendering && clip.urls && (
            <div className="absolute inset-x-0 bottom-0 bg-black/85 p-3 text-[13px] text-muted">
              Rendering a new version
              <div className="mt-2">
                <ProgressBar value={clip.progress} label="Render progress" indeterminate={clip.status === "queued"} />
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="min-w-0">
        <div className="flex flex-wrap items-start justify-between gap-x-8 gap-y-4">
          <div>
            <p className="text-sm text-muted">
              Clip {clip.rank}
              {clip.variant && <span className="ml-2 rounded-md border border-line px-1.5 py-0.5 text-[12px] text-fg">Version {clip.variant.index}: {clip.variant.label}</span>}
            </p>
            <ScoreNumber clip={clip} />
            {clip.applied_style && <p className="mt-1 text-[13px] text-faint">Style: {clip.applied_style.label}</p>}
            {clip.variant && <p className="mt-1 text-[13px] text-faint">{clip.variant.why}</p>}
            {clip.recommendation && clip.applied_style && clip.recommendation.style !== clip.applied_style.style && (
              <p className="mt-1 text-[13px] text-faint">
                Suggested: {clip.recommendation.label} ({clip.recommendation.reasons[0]}). Open Edit to try it.
              </p>
            )}
          </div>
          <div className="flex flex-wrap gap-2">
            {clip.urls && (
              <a href={mediaUrl(clip.urls.download)} className={buttonClass("primary", "md")}>
                <Download className="size-4" aria-hidden />
                Download MP4
              </a>
            )}
            {clip.urls && clip.has_srt && (
              <a href={mediaUrl(clip.urls.srt)} className={buttonClass("secondary", "md")}>
                <Captions className="size-4" aria-hidden />
                SRT
              </a>
            )}
            <Link href={`/jobs/${jobId}/clips/${clip.id}`} className={buttonClass("secondary", "md")}>
              <PenLine className="size-4" aria-hidden />
              Edit
            </Link>
          </div>
        </div>

        {clip.status === "failed" && (
          <div className="mt-4">
            <Notice tone="error">{clip.error || "This clip could not be rendered."} Open Edit to adjust it and render again.</Notice>
          </div>
        )}
        {clip.error && clip.status === "ready" && (
          <div className="mt-4">
            <Notice tone="warn">{clip.error}</Notice>
          </div>
        )}

        <p className="mt-4 max-w-2xl text-[15px]">{clip.explanation}</p>
        {clip.score_stale && <p className="mt-1 text-[13px] text-faint">You moved the in or out point, so this score still describes the original selection.</p>}

        {clip.hook.text && (
          <figure className="mt-5 max-w-2xl border-l-2 border-amber pl-4">
            <blockquote className="font-display text-xl font-medium leading-snug">“{clip.hook.text}”</blockquote>
            <figcaption className="mt-1.5 text-[13px] text-muted">The opening{hookLabels ? `, identified as ${hookLabels}` : ""}</figcaption>
          </figure>
        )}

        <dl className="tnum mt-6 flex flex-wrap gap-x-8 gap-y-2 text-sm">
          <div>
            <dt className="text-faint">Length</dt>
            <dd>{seconds(clip.duration)}</dd>
          </div>
          <div>
            <dt className="text-faint">From the source</dt>
            <dd>
              {timecode(clip.start)} to {timecode(clip.end)}
            </dd>
          </div>
          {clip.language && (
            <div>
              <dt className="text-faint">Language</dt>
              <dd>{languageName(clip.language)}</dd>
            </div>
          )}
          {clip.layouts.length > 0 && (
            <div>
              <dt className="text-faint">Framing</dt>
              <dd>{clip.layouts.map(layoutName).join(" + ")}</dd>
            </div>
          )}
          <div>
            <dt className="text-faint">Captions</dt>
            <dd>{clip.has_captions_burned ? "Burned in" : "None on picture"}</dd>
          </div>
          {clip.qa && (
            <div>
              <dt className="text-faint">File</dt>
              <dd>
                {clip.qa.width} × {clip.qa.height}, checked
              </dd>
            </div>
          )}
        </dl>

        {clip.titles.length > 0 && (
          <div className="mt-7 max-w-2xl">
            <h3 className="font-sans text-sm font-medium tracking-normal text-muted">
              {clip.copy_source === "llm" ? "Title ideas, written from the transcript" : "Title ideas, quoted from the clip"}
            </h3>
            <ul className="mt-2 divide-y divide-line border-y border-line">
              {clip.titles.map((t) => (
                <li key={t} className="flex items-center justify-between gap-3 py-2">
                  <span className="text-[15px]">{t}</span>
                  <CopyButton text={t} label="Copy title" />
                </li>
              ))}
            </ul>
          </div>
        )}

        {clip.caption && (
          <div className="mt-6 max-w-2xl">
            <div className="flex items-center justify-between">
              <h3 className="font-sans text-sm font-medium tracking-normal text-muted">Caption to post with</h3>
              <CopyButton text={clip.caption} label="Copy caption" />
            </div>
            <p className="mt-1 whitespace-pre-line text-[15px]">{clip.caption}</p>
          </div>
        )}

        <details className="mt-6 max-w-2xl">
          <summary className="text-sm text-muted hover:text-fg">How this score was built</summary>
          <div className="mt-4">
            <ScoreBreakdown clip={clip} />
          </div>
        </details>
      </div>
    </li>
  );
}

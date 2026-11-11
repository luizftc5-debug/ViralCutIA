"use client";

import { FileArchive } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { ClipRow } from "@/components/clip-row";
import { SiteHeader } from "@/components/site-header";
import { TranscriptReview } from "@/components/transcript-review";
import { JobLog, StageList } from "@/components/stage-list";
import { Button, buttonClass, Notice, ProgressBar } from "@/components/ui";
import { api, DEFAULT_OPTIONS, exportUrl, type Job, type JobOptions } from "@/lib/api";
import { languageName, timecode } from "@/lib/format";
import { forgetJob, rememberJob } from "@/lib/recent";
import { useJob } from "@/lib/use-job";

const ENGINE_SUMMARY = {
  heuristic: "ranked from speech, audio and picture signals",
  "llm+signals": "ranked by a language model reading the transcript, with audio and picture signals",
  "signals-only": "ranked from audio and picture signals only",
};

function currentOptions(job: Job): JobOptions {
  return { ...DEFAULT_OPTIONS, ...job.options, captions: { ...DEFAULT_OPTIONS.captions, ...job.options.captions } };
}

export default function JobPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();
  const { job, error, refresh } = useJob(id);
  const [showLog, setShowLog] = useState(false);
  const [actionError, setActionError] = useState("");
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (job) rememberJob({ id: job.id, filename: job.filename, createdAt: job.created_at });
  }, [job]);
  useEffect(() => {
    if (error?.notFound) forgetJob(id);
  }, [error, id]);

  const act = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setActionError("");
    try {
      await fn();
      refresh();
    } catch (e) {
      setActionError(e instanceof Error ? e.message : "That didn’t work. Try again.");
    } finally {
      setBusy(false);
    }
  };

  if (error?.notFound) {
    return (
      <>
        <SiteHeader studio />
        <main id="main" className="mx-auto max-w-[1240px] px-5 py-20 sm:px-8">
          <h1 className="text-4xl font-semibold">This video is no longer here</h1>
          <p className="mt-4 max-w-xl text-muted">Uploads and clips are deleted automatically when the retention period ends, or the link may be mistyped. Upload the video again to make new clips.</p>
          <Link href="/new" className={`${buttonClass("primary", "lg")} mt-8`}>
            Upload a video
          </Link>
        </main>
      </>
    );
  }

  if (!job) {
    return (
      <>
        <SiteHeader studio />
        <main id="main" className="mx-auto max-w-[1240px] px-5 py-20 sm:px-8">
          {error ? <Notice tone="error">{error.message}</Notice> : <p className="text-muted">Loading…</p>}
        </main>
      </>
    );
  }

  const active = job.status === "queued" || job.status === "processing";
  const a = job.analysis;
  const ready = job.clips.filter((c) => c.status === "ready");
  const rerun = (patch: Partial<JobOptions> = {}) => act(() => api.start(job.id, { ...currentOptions(job), ...patch }));

  return (
    <>
      <SiteHeader studio />
      <main id="main" className="mx-auto max-w-[1240px] px-5 pb-24 pt-10 sm:px-8">
        <p className="tnum text-sm text-muted">
          <span className="text-fg">{job.filename}</span>
          {job.media.duration ? `, ${timecode(job.media.duration)}` : ""}
        </p>

        {error && !error.notFound && (
          <div className="mt-4">
            <Notice tone="warn">{error.message} Retrying…</Notice>
          </div>
        )}
        {actionError && (
          <div className="mt-4">
            <Notice tone="error">{actionError}</Notice>
          </div>
        )}

        {/* ------------------------------------------------ processing */}
        {active && (
          <section className="mt-3 max-w-2xl" aria-live="polite">
            <h1 className="text-4xl font-semibold">{job.status === "queued" ? "Waiting for a free worker" : "Finding your best moments"}</h1>
            <div className="mt-6 flex items-center gap-4">
              <div className="flex-1">
                <ProgressBar value={job.progress} label="Overall progress" indeterminate={job.status === "queued"} />
              </div>
              <span className="tnum w-12 text-right text-sm text-muted">{Math.round(job.progress * 100)}%</span>
            </div>
            <div className="mt-8">
              <StageList job={job} />
            </div>
            <div className="mt-6 flex flex-wrap items-center gap-3">
              <Button variant="danger" disabled={busy || job.cancel_requested} onClick={() => act(() => api.cancel(job.id))}>
                {job.cancel_requested ? "Canceling…" : "Cancel"}
              </Button>
              <Button variant="ghost" aria-expanded={showLog} onClick={() => setShowLog((v) => !v)}>
                {showLog ? "Hide log" : "Show log"}
              </Button>
            </div>
            <p className="mt-4 text-sm text-faint">Processing runs on the server. You can close this page and return to this link later.</p>
            {showLog && (
              <div className="mt-5">
                <JobLog jobId={job.id} live />
              </div>
            )}
          </section>
        )}

        {/* ------------------------------------------------ failed / canceled / waiting */}
        {job.status === "failed" && (
          <section className="mt-3 max-w-2xl">
            <h1 className="text-4xl font-semibold">Processing stopped</h1>
            <div className="mt-6">
              <Notice tone="error">{job.error || "Something went wrong while processing this video."}</Notice>
            </div>
            <div className="mt-6 flex gap-3">
              <Button variant="primary" disabled={busy} onClick={() => rerun()}>
                Try again
              </Button>
              <Button variant="ghost" aria-expanded={showLog} onClick={() => setShowLog((v) => !v)}>
                {showLog ? "Hide log" : "Show log"}
              </Button>
            </div>
            {showLog && (
              <div className="mt-5">
                <JobLog jobId={job.id} live={false} />
              </div>
            )}
          </section>
        )}
        {job.status === "canceled" && (
          <section className="mt-3 max-w-2xl">
            <h1 className="text-4xl font-semibold">Canceled</h1>
            <p className="mt-4 text-muted">Nothing was rendered. Your upload is still here if you want to run it again.</p>
            <div className="mt-6 flex gap-3">
              <Button variant="primary" disabled={busy} onClick={() => rerun()}>
                Generate clips
              </Button>
              <Button
                variant="danger"
                disabled={busy}
                onClick={() =>
                  act(async () => {
                    await api.remove(job.id);
                    forgetJob(job.id);
                    router.push("/new");
                  })
                }
              >
                Delete this upload
              </Button>
            </div>
          </section>
        )}
        {job.status === "review" && (
          <section className="mt-3">
            <h1 className="text-4xl font-semibold">Check the transcript</h1>
            <p className="mt-4 max-w-2xl text-muted">
              {job.clips.length} clip{job.clips.length === 1 ? "" : "s"} chosen. Fix any misheard words now and they are corrected in every caption, then continue to render.
            </p>
            <TranscriptReview jobId={job.id} busy={busy} onContinue={() => act(() => api.resume(job.id))} />
          </section>
        )}
        {job.status === "uploaded" && (
          <section className="mt-3 max-w-2xl">
            <h1 className="text-4xl font-semibold">Uploaded, not processed yet</h1>
            <p className="mt-4 text-muted">This video is on the server but clips were never generated.</p>
            <Button variant="primary" className="mt-6" disabled={busy} onClick={() => rerun()}>
              Generate clips with default settings
            </Button>
          </section>
        )}

        {/* ------------------------------------------------ results */}
        {job.status === "completed" && (
          <section className="mt-3">
            <div className="flex flex-wrap items-end justify-between gap-6">
              <div>
                <h1 className="text-4xl font-semibold">
                  {job.clips.length === 0 ? "No moment made the cut" : job.clips.length === 1 ? "1 clip, ready to publish" : `${job.clips.length} clips, ranked`}
                </h1>
                {job.clips.length > 0 && (
                  <p className="mt-3 max-w-2xl text-muted">
                    {a.language ? `${languageName(a.language)} detected. ` : ""}
                    {a.candidates ? `${a.candidates.toLocaleString()} candidate moments scored, ` : ""}
                    {a.engine ? `${ENGINE_SUMMARY[a.engine]}.` : ""}
                    {a.requested && a.selected !== undefined && a.selected < a.requested ? ` Up to ${a.requested} were allowed; only ${a.selected} reached the minimum score of ${Math.round(a.min_score ?? 0)}.` : ""}
                  </p>
                )}
              </div>
              {ready.length > 0 && (
                <a href={exportUrl(job.id)} className={buttonClass("secondary", "md")}>
                  <FileArchive className="size-4" aria-hidden />
                  Download all as ZIP
                </a>
              )}
            </div>

            {(a.warnings ?? []).length > 0 && (
              <div className="mt-6 max-w-3xl space-y-2">
                {(a.warnings ?? []).map((w) => (
                  <Notice key={w} tone="warn">
                    {w}
                  </Notice>
                ))}
              </div>
            )}

            {job.clips.length === 0 ? (
              <div className="mt-6 max-w-2xl">
                <p className="text-muted">
                  The strongest moment found scored {Math.round(a.best_candidate_score ?? 0)}, below the minimum of {Math.round(a.min_score ?? 0)}. Rather than hand you weak clips,
                  nothing was rendered.
                </p>
                <div className="mt-6 flex flex-wrap gap-3">
                  <Button variant="primary" disabled={busy} onClick={() => rerun({ min_score: Math.max(0, Math.floor((a.best_candidate_score ?? 40) - 5)) })}>
                    Lower the minimum and run again
                  </Button>
                  <Link href="/new" className={buttonClass("secondary", "md")}>
                    Try another video
                  </Link>
                </div>
              </div>
            ) : (
              <ul className="mt-4 divide-y divide-line border-b border-line">
                {job.clips.map((c) => (
                  <ClipRow key={c.id} clip={c} jobId={job.id} />
                ))}
              </ul>
            )}

            <div className="mt-8 flex flex-wrap items-center gap-x-6 gap-y-3 text-sm text-faint">
              {job.expires_at && <p>Files are deleted automatically on {new Date(job.expires_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}.</p>}
              <button type="button" className="text-muted underline-offset-4 hover:text-fg hover:underline" aria-expanded={showLog} onClick={() => setShowLog((v) => !v)}>
                {showLog ? "Hide processing log" : "Show processing log"}
              </button>
              <button
                type="button"
                className="text-danger underline-offset-4 hover:underline"
                disabled={busy}
                onClick={() => {
                  if (!window.confirm("Delete this video and all of its clips from the server? This can’t be undone.")) return;
                  void act(async () => {
                    await api.remove(job.id);
                    forgetJob(job.id);
                    router.push("/new");
                  });
                }}
              >
                Delete now
              </button>
            </div>
            {showLog && (
              <div className="mt-5 max-w-3xl">
                <JobLog jobId={job.id} live={false} />
              </div>
            )}
          </section>
        )}
      </main>
    </>
  );
}

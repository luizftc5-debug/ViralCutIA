"use client";

import { Trash2 } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { api, type JobSummary, mediaUrl } from "@/lib/api";
import { timecode } from "@/lib/format";
import { forgetJob } from "@/lib/recent";
import { isActive, projectStatus, type StatusTone } from "@/lib/status";

import { cx, ProgressBar } from "./ui";

const DOT: Record<StatusTone, string> = { ok: "bg-ok", busy: "bg-amber", bad: "bg-danger", idle: "bg-line-strong" };

/** Every video on this installation, newest first; refreshed while one of them is being processed. */
export function useProjects() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState("");
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const { jobs: next } = await api.jobs();
        if (!alive) return;
        setJobs(next);
        setError("");
        if (next.some(isActive)) timer = setTimeout(tick, 3000);
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : "Could not load your videos.");
      }
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return { jobs, error, reload };
}

export function ProjectCard({ job, onChanged }: { job: JobSummary; onChanged: () => void }) {
  const status = projectStatus(job);
  const [busy, setBusy] = useState(false);
  const [problem, setProblem] = useState("");

  const remove = async () => {
    if (!window.confirm(`Delete “${job.filename}” and all of its clips? This can’t be undone.`)) return;
    setBusy(true);
    setProblem("");
    try {
      await api.remove(job.id);
      forgetJob(job.id);
      onChanged();
    } catch (e) {
      setProblem(e instanceof Error ? e.message : "Could not delete this video.");
      setBusy(false);
    }
  };

  return (
    <li className={cx("relative rounded-xl border border-line bg-panel transition-colors hover:border-line-strong", busy && "opacity-50")}>
      <Link href={`/jobs/${job.id}`} className="flex gap-4 p-3 pr-12">
        <div className="h-28 w-16 shrink-0 overflow-hidden rounded-md bg-black">
          {job.thumbnail && (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={mediaUrl(job.thumbnail)} alt="" loading="lazy" className="size-full object-cover" />
          )}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium" title={job.filename}>
            {job.filename}
          </p>
          <p className="mt-1.5 flex items-center gap-2 text-[13px] text-fg">
            <span aria-hidden className={cx("size-2 shrink-0 rounded-full", DOT[status.tone], job.status === "processing" && "animate-pulse")} />
            <span className="truncate">{status.text}</span>
          </p>
          {job.status === "processing" && (
            <div className="mt-2">
              <ProgressBar value={job.progress} label={`Progress of ${job.filename}`} />
            </div>
          )}
          <p className="tnum mt-2 text-[12px] text-faint">
            {job.media?.duration ? `${timecode(job.media.duration)} · ` : ""}
            {new Date(job.created_at).toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" })}
          </p>
          {problem && (
            <p role="alert" className="mt-1 text-[12px] text-danger">
              {problem}
            </p>
          )}
        </div>
      </Link>
      {job.status !== "processing" && (
        <button
          type="button"
          onClick={remove}
          disabled={busy}
          aria-label={`Delete ${job.filename}`}
          title="Delete this video"
          className="absolute right-2 top-2 inline-flex size-8 items-center justify-center rounded-md text-faint transition-colors hover:bg-hover hover:text-danger"
        >
          <Trash2 className="size-4" aria-hidden />
        </button>
      )}
    </li>
  );
}

export function ProjectGrid({ jobs, onChanged }: { jobs: JobSummary[]; onChanged: () => void }) {
  return (
    <ul className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      {jobs.map((j) => (
        <ProjectCard key={j.id} job={j} onChanged={onChanged} />
      ))}
    </ul>
  );
}

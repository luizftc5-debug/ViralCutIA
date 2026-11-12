"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { SiteHeader } from "@/components/site-header";
import { Notice } from "@/components/ui";
import { api, type JobSummary, mediaUrl } from "@/lib/api";
import { timecode } from "@/lib/format";

const STATUS: Record<string, string> = {
  uploaded: "Uploaded",
  queued: "Waiting",
  processing: "Processing",
  review: "Waiting for transcript review",
  completed: "Ready",
  failed: "Failed",
  canceled: "Canceled",
};

/** Project history: every video processed on this machine, newest first (until retention deletes it). */
export default function ProjectsPage() {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    api
      .jobs()
      .then((r) => setJobs(r.jobs))
      .catch((e: unknown) => setError(e instanceof Error ? e.message : "Could not load projects."));
  }, []);
  return (
    <>
      <SiteHeader studio />
      <main id="main" className="mx-auto max-w-[1240px] px-5 py-12 sm:px-8">
        <h1 className="text-3xl font-semibold">Projects</h1>
        <p className="mt-2 text-sm text-muted">Reopen a video to download, compare or re-edit its clips. Files are kept until the retention period ends.</p>
        {error && (
          <div className="mt-6">
            <Notice tone="error">{error}</Notice>
          </div>
        )}
        {jobs && jobs.length === 0 && <p className="mt-10 text-muted">No projects yet.</p>}
        <ul className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {(jobs ?? []).map((j) => (
            <li key={j.id}>
              <Link href={`/jobs/${j.id}`} className="flex gap-4 rounded-xl border border-line bg-panel p-3 hover:border-line-strong">
                <div className="h-28 w-16 shrink-0 overflow-hidden rounded-md bg-black">
                  {j.thumbnail && (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img src={mediaUrl(j.thumbnail)} alt="" loading="lazy" className="size-full object-cover" />
                  )}
                </div>
                <div className="min-w-0">
                  <p className="truncate font-medium" title={j.filename}>
                    {j.filename}
                  </p>
                  <p className="mt-1 text-[13px] text-faint">
                    {STATUS[j.status] ?? j.status}
                    {j.media?.duration ? ` · ${timecode(j.media.duration)}` : ""}
                  </p>
                  <p className="mt-1 text-[13px] text-faint">
                    {j.clip_count} clip{j.clip_count === 1 ? "" : "s"}
                    {j.options?.preset ? ` · ${j.options.preset}` : ""}
                    {j.options?.content_type ? ` · ${j.options.content_type}` : ""}
                  </p>
                  <p className="mt-1 text-[12px] text-faint">{new Date(j.created_at).toLocaleString()}</p>
                </div>
              </Link>
            </li>
          ))}
        </ul>
      </main>
    </>
  );
}

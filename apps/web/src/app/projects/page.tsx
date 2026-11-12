"use client";

import Link from "next/link";

import { ProjectGrid, useProjects } from "@/components/project-list";
import { SiteHeader } from "@/components/site-header";
import { buttonClass, Notice } from "@/components/ui";

/** Project history: every video processed on this machine, newest first (until retention deletes it). */
export default function ProjectsPage() {
  const { jobs, error, reload } = useProjects();
  return (
    <>
      <SiteHeader studio />
      <main id="main" className="mx-auto max-w-[1240px] px-5 py-12 sm:px-8">
        <h1 className="text-3xl font-semibold">Projects</h1>
        <p className="mt-2 text-sm text-muted">Open a video to follow its progress, download its clips or edit them. Files are deleted automatically when the retention period ends.</p>
        {error && (
          <div className="mt-6">
            <Notice tone="error">{error}</Notice>
          </div>
        )}
        {!jobs && !error && <p className="mt-10 text-faint">Loading…</p>}
        {jobs && jobs.length === 0 && (
          <div className="mt-10 max-w-md">
            <p className="text-muted">No videos yet. Upload one and its clips will be listed here.</p>
            <Link href="/new" className={`${buttonClass("primary", "md")} mt-5`}>
              Upload a video
            </Link>
          </div>
        )}
        {jobs && jobs.length > 0 && (
          <div className="mt-8">
            <ProjectGrid jobs={jobs} onChanged={reload} />
          </div>
        )}
      </main>
    </>
  );
}

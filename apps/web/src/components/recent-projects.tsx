"use client";

import Link from "next/link";

import { ProjectGrid, useProjects } from "./project-list";

/** The latest videos, right under the upload: what is running, what is ready, what needs a look. */
export function RecentProjects({ limit = 3 }: { limit?: number }) {
  const { jobs, reload } = useProjects();
  if (!jobs || jobs.length === 0) return null; // nothing yet, or the processing server is not reachable
  return (
    <section aria-labelledby="recent-heading" className="mt-12">
      <div className="mb-4 flex items-baseline justify-between gap-4">
        <h2 id="recent-heading" className="text-lg font-semibold">
          Your videos
        </h2>
        {jobs.length > limit && (
          <Link href="/projects" className="text-sm text-muted underline-offset-4 hover:text-fg hover:underline">
            See all {jobs.length}
          </Link>
        )}
      </div>
      <ProjectGrid jobs={jobs.slice(0, limit)} onChanged={reload} />
    </section>
  );
}

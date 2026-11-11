"use client";

import Link from "next/link";
import { useSyncExternalStore } from "react";

import { listRecent, type RecentJob } from "@/lib/recent";

let cache: { raw: string; items: RecentJob[] } = { raw: "", items: [] };
const EMPTY: RecentJob[] = [];

function snapshot(): RecentJob[] {
  const items = listRecent();
  const raw = JSON.stringify(items);
  if (raw !== cache.raw) cache = { raw, items };
  return cache.items;
}

function subscribe(cb: () => void): () => void {
  window.addEventListener("viralcut:recent", cb);
  window.addEventListener("storage", cb);
  return () => {
    window.removeEventListener("viralcut:recent", cb);
    window.removeEventListener("storage", cb);
  };
}

/** Projects opened in this browser. They stay on the server only until the retention period ends. */
export function RecentProjects() {
  const items = useSyncExternalStore(subscribe, snapshot, () => EMPTY);
  if (items.length === 0) return null;
  return (
    <section aria-labelledby="recent-heading" className="mt-12">
      <h2 id="recent-heading" className="text-lg font-semibold">
        Recent videos on this device
      </h2>
      <ul className="mt-3 divide-y divide-line border-y border-line">
        {items.slice(0, 6).map((j) => (
          <li key={j.id}>
            <Link href={`/jobs/${j.id}`} className="flex items-center justify-between gap-4 py-3 text-sm hover:text-amber">
              <span className="truncate">{j.filename}</span>
              <time dateTime={j.createdAt} className="tnum shrink-0 text-faint">
                {new Date(j.createdAt).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
              </time>
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

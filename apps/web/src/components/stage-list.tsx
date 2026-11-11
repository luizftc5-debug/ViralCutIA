"use client";

import { Check, Minus } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api, type Job, type LogEntry } from "@/lib/api";

import { cx, ProgressBar } from "./ui";

/** The eight pipeline stages with the progress the worker actually reported for each. */
export function StageList({ job }: { job: Job }) {
  return (
    <ol className="divide-y divide-line border-y border-line">
      {job.stages.map((s, i) => {
        const active = s.status === "active";
        return (
          <li key={s.key} aria-current={active ? "step" : undefined} className="grid grid-cols-[2rem_1fr_auto] items-center gap-x-3 py-3.5">
            <span
              className={cx(
                "tnum flex size-6 items-center justify-center rounded-full text-[12px]",
                s.status === "done" && "bg-amber text-black",
                active && "border border-amber text-amber",
                s.status === "pending" && "border border-line text-faint",
                s.status === "skipped" && "border border-line text-faint",
              )}
            >
              {s.status === "done" ? <Check className="size-3.5" aria-hidden /> : s.status === "skipped" ? <Minus className="size-3.5" aria-hidden /> : i + 1}
            </span>
            <span className={cx("text-[15px]", s.status === "pending" || s.status === "skipped" ? "text-faint" : "text-fg")}>
              {s.label}
              {s.status === "skipped" && <span className="ml-2 text-[13px]">not needed for this video</span>}
            </span>
            <span className="tnum text-sm text-muted">
              {active && s.key !== "ready" ? `${Math.round(s.progress * 100)}%` : ""}
              <span className="sr-only">{s.status === "done" ? "done" : s.status === "pending" ? "waiting" : ""}</span>
            </span>
            {active && s.key !== "ready" && (
              <div className="col-span-2 col-start-2 mt-2.5">
                <ProgressBar value={s.progress} label={`${s.label} progress`} />
              </div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

export function JobLog({ jobId, live }: { jobId: string; live: boolean }) {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const last = useRef(0);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      try {
        const { logs: fresh } = await api.logs(jobId, last.current);
        const newest = fresh.at(-1);
        if (!stop && newest) {
          last.current = newest.id;
          setLogs((prev) => [...prev, ...fresh]);
        }
      } catch {
        /* transient: next tick retries */
      }
      if (!stop && live) timer = setTimeout(tick, 1500);
    };
    void tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [jobId, live]);

  useEffect(() => {
    box.current?.scrollTo({ top: box.current.scrollHeight });
  }, [logs.length]);

  return (
    <div ref={box} tabIndex={0} role="log" aria-label="Processing log" className="max-h-72 overflow-y-auto rounded-lg border border-line bg-panel p-4 text-[13px] leading-relaxed">
      {logs.length === 0 && <p className="text-faint">Nothing logged yet.</p>}
      {logs.map((l) => (
        <p key={l.id} className={cx(l.level === "error" ? "text-danger" : l.level === "warn" ? "text-amber" : "text-muted")}>
          <time className="tnum mr-3 text-faint">{new Date(l.ts).toLocaleTimeString(undefined, { hour12: false })}</time>
          {l.message}
        </p>
      ))}
    </div>
  );
}

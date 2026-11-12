"use client";

import { Check, Minus } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import { api, type Job, type LogEntry, type Stage } from "@/lib/api";
import { elapsed } from "@/lib/format";

import { cx, ProgressBar } from "./ui";

/** Seconds since `since`, ticking once a second while `running` (null when the time is unknown). */
export function useElapsed(since: string | null | undefined, running: boolean): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, [running]);
  if (!since) return null;
  const start = Date.parse(since);
  return Number.isFinite(start) ? Math.max(0, (now - start) / 1000) : null;
}

function StageRow({ stage: s, index }: { stage: Stage; index: number }) {
  const active = s.status === "active";
  const running = useElapsed(s.started_at, active);
  // a step that has not reported progress yet (a download, loading a model) is working, not stuck:
  // it gets a moving bar and says what it is doing instead of a still "0%"
  const silent = active && s.progress <= 0;
  return (
    <li aria-current={active ? "step" : undefined} className="grid grid-cols-[2rem_1fr_auto] items-center gap-x-3 py-3.5">
      <span
        className={cx(
          "tnum flex size-6 items-center justify-center rounded-full text-[12px]",
          s.status === "done" && "bg-amber text-black",
          active && "border border-amber text-amber",
          s.status === "pending" && "border border-line text-faint",
          s.status === "skipped" && "border border-line text-faint",
        )}
      >
        {s.status === "done" ? <Check className="size-3.5" aria-hidden /> : s.status === "skipped" ? <Minus className="size-3.5" aria-hidden /> : index + 1}
      </span>
      <span className={cx("text-[15px]", s.status === "pending" || s.status === "skipped" ? "text-faint" : "text-fg")}>
        {s.label}
        {s.status === "skipped" && <span className="ml-2 text-[13px]">not needed for this video</span>}
      </span>
      <span className="tnum text-sm text-muted">
        {active && s.key !== "ready" && !silent ? `${Math.round(s.progress * 100)}%` : ""}
        {active && running != null && running >= 1 ? `${silent ? "" : " · "}${elapsed(running)}` : ""}
        {s.status === "done" && s.elapsed_s != null && s.elapsed_s >= 0.1 ? (s.elapsed_s < 60 ? `${s.elapsed_s.toFixed(1)} s` : elapsed(s.elapsed_s)) : ""}
        <span className="sr-only">{s.status === "done" ? "done" : s.status === "pending" ? "waiting" : ""}</span>
      </span>
      {active && s.key !== "ready" && (
        <div className="col-span-2 col-start-2 mt-2">
          {s.detail && <p className="mb-2 text-[13px] text-muted">{s.detail}</p>}
          <ProgressBar value={s.progress} label={`${s.label} progress`} indeterminate={silent} />
        </div>
      )}
    </li>
  );
}

/** The pipeline stages with the progress the worker actually reported for each. */
export function StageList({ job }: { job: Job }) {
  return (
    <ol className="divide-y divide-line border-y border-line">
      {job.stages.map((s, i) => (
        <StageRow key={s.key} stage={s} index={i} />
      ))}
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
        <p key={l.id} className={cx("break-words", l.level === "error" ? "text-danger" : l.level === "warn" ? "text-amber" : "text-muted")}>
          <time className="tnum mr-3 text-faint">{new Date(l.ts).toLocaleTimeString(undefined, { hour12: false })}</time>
          {l.message}
        </p>
      ))}
    </div>
  );
}

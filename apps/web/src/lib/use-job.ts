"use client";

import { useCallback, useEffect, useState } from "react";

import { api, ApiError, type Job } from "./api";

const ACTIVE = new Set(["queued", "processing"]);

export function isBusy(job: Job | null): boolean {
  if (!job) return false;
  return ACTIVE.has(job.status) || job.clips.some((c) => c.status === "queued" || c.status === "rendering");
}

/** Polls a job while work is in progress and stops once everything is settled. */
export function useJob(id: string) {
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState<{ message: string; notFound: boolean } | null>(null);
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const next = await api.job(id);
        if (!alive) return;
        setJob(next);
        setError(null);
        if (isBusy(next)) timer = setTimeout(tick, 1000);
      } catch (e) {
        if (!alive) return;
        const notFound = e instanceof ApiError && e.status === 404;
        setError({ message: e instanceof Error ? e.message : "Could not load this video.", notFound });
        if (!notFound) timer = setTimeout(tick, 4000); // server briefly unreachable: keep trying
      }
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return { job, error, refresh };
}

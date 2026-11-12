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
        // a background tab does not need second-by-second updates: poll 5x less often
        if (isBusy(next)) timer = setTimeout(tick, document.visibilityState === "hidden" ? 5000 : 1000);
      } catch (e) {
        if (!alive) return;
        const notFound = e instanceof ApiError && e.status === 404;
        setError({ message: e instanceof Error ? e.message : "Could not load this video.", notFound });
        if (!notFound) timer = setTimeout(tick, 4000); // server briefly unreachable: keep trying
      }
    };
    void tick();
    const onVisible = () => {
      if (document.visibilityState === "visible" && alive) {
        clearTimeout(timer);
        void tick(); // catch up immediately when the user comes back
      }
    };
    document.addEventListener("visibilitychange", onVisible);
    return () => {
      alive = false;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVisible);
    };
  }, [id, nonce]);

  const refresh = useCallback(() => setNonce((n) => n + 1), []);
  return { job, error, refresh };
}

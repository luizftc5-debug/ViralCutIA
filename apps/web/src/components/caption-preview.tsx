"use client";

import { useEffect, useRef, useState } from "react";

import { api, type CaptionOptions, type JobOptions } from "@/lib/api";

/**
 * Live caption preview. The frame is rendered on the server by the export engine itself (same
 * layout code, same libass), so it matches the MP4. Requests are debounced, the previous one is
 * aborted, and every object URL is revoked, so tweaking sliders never piles up work or memory.
 */
export function CaptionPreview({
  captions,
  platform,
  jobId,
  clipId,
  language,
}: {
  captions: CaptionOptions;
  platform?: JobOptions["platform"];
  jobId?: string;
  clipId?: string;
  language?: string;
}) {
  const [url, setUrl] = useState<string>("");
  const [result, setResult] = useState<{ key: string; ok: boolean }>({ key: "", ok: true });
  const urlRef = useRef("");
  const key = JSON.stringify({ captions, platform, jobId, clipId, language });
  const state = result.key !== key ? "loading" : result.ok ? "ready" : "error";

  useEffect(() => {
    const ctrl = new AbortController();
    const timer = window.setTimeout(async () => {
      try {
        const blob = await api.captionPreview({ captions: { ...captions, enabled: true }, platform, job_id: jobId, clip_id: clipId, language }, ctrl.signal);
        const next = URL.createObjectURL(blob);
        if (urlRef.current) URL.revokeObjectURL(urlRef.current);
        urlRef.current = next;
        setUrl(next);
        setResult({ key, ok: true });
      } catch (e) {
        if ((e as Error).name !== "AbortError") setResult({ key, ok: false });
      }
    }, 350);
    return () => {
      window.clearTimeout(timer);
      ctrl.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(
    () => () => {
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    },
    [],
  );

  return (
    <figure className="w-full max-w-[240px]">
      <div className="relative aspect-[9/16] overflow-hidden rounded-xl border border-line bg-panel">
        {url && (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={url} alt="Caption preview rendered by the export engine" className={state === "loading" ? "size-full object-cover opacity-60" : "size-full object-cover"} />
        )}
        {state === "error" && <p className="absolute inset-0 grid place-items-center p-4 text-center text-[13px] text-faint">Preview unavailable. Check the server.</p>}
      </div>
      <figcaption className="mt-2 text-[12px] text-faint">Rendered by the same engine as the export.</figcaption>
    </figure>
  );
}

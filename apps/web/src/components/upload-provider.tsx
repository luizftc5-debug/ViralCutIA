"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";

import { api, type AppConfig, type Job, type UploadHandle, uploadVideo } from "@/lib/api";
import { rememberJob } from "@/lib/recent";

export type UploadPhase = "idle" | "uploading" | "ready" | "error";

interface UploadState {
  phase: UploadPhase;
  file: File | null;
  progress: number;
  job: Job | null;
  error: string;
  config: AppConfig | null;
  /** Validate and start uploading. Returns an error message, or null when the upload started. */
  begin: (file: File) => string | null;
  reset: () => void;
}

const FALLBACK_FORMATS = ["MP4", "MOV", "MKV", "WEBM"];

const Ctx = createContext<UploadState | null>(null);

/** Holds the in-flight upload above the router so it survives the move from landing to studio. */
export function UploadProvider({ children }: { children: React.ReactNode }) {
  const [phase, setPhase] = useState<UploadPhase>("idle");
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [job, setJob] = useState<Job | null>(null);
  const [error, setError] = useState("");
  const [config, setConfig] = useState<AppConfig | null>(null);
  const handle = useRef<UploadHandle | null>(null);

  useEffect(() => {
    api.config().then(setConfig).catch(() => setConfig(null));
  }, []);

  const reset = useCallback(() => {
    handle.current?.abort();
    handle.current = null;
    setPhase("idle");
    setFile(null);
    setProgress(0);
    setJob(null);
    setError("");
  }, []);

  const begin = useCallback(
    (next: File): string | null => {
      const formats = config?.limits.formats ?? FALLBACK_FORMATS;
      const ext = next.name.split(".").pop()?.toUpperCase() ?? "";
      if (!formats.includes(ext)) {
        return `${ext ? `.${ext.toLowerCase()} files are` : "This file type is"} not supported. Use ${formats.join(", ")}.`;
      }
      const maxMb = config?.limits.max_upload_mb;
      if (maxMb && next.size > maxMb * 1024 * 1024) {
        return `This file is larger than the ${maxMb >= 1024 ? `${(maxMb / 1024).toFixed(0)} GB` : `${maxMb} MB`} limit.`;
      }
      handle.current?.abort();
      setFile(next);
      setJob(null);
      setError("");
      setProgress(0);
      setPhase("uploading");
      const h = uploadVideo(next, setProgress);
      handle.current = h;
      h.promise
        .then((created) => {
          if (handle.current !== h) return;
          setJob(created);
          setPhase("ready");
          rememberJob({ id: created.id, filename: created.filename, createdAt: created.created_at });
        })
        .catch((e: { status?: number; message: string }) => {
          if (handle.current !== h || e.status === -1) return;
          setError(e.message);
          setPhase("error");
        });
      return null;
    },
    [config],
  );

  const value = useMemo(
    () => ({ phase, file, progress, job, error, config, begin, reset }),
    [phase, file, progress, job, error, config, begin, reset],
  );
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useUpload(): UploadState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useUpload must be used inside <UploadProvider>");
  return ctx;
}

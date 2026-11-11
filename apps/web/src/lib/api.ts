/** Typed client for the ViralCut API. The browser talks to the API directly. */

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type CaptionStyle = "high_retention" | "clean_editorial" | "dynamic_creator" | "gaming";
export type FramingMode = "auto" | "face" | "split" | "blur" | "center";

export interface CaptionOptions {
  enabled: boolean;
  style: CaptionStyle;
  highlight_keywords: boolean;
  size: "s" | "m" | "l";
  position: "top" | "middle" | "bottom";
  color: string;
  font: "auto" | "inter" | "poppins" | "anton" | "dm_serif";
  animation: boolean;
}

export interface JobOptions {
  captions: CaptionOptions;
  language: string;
  clip_count: number | "auto";
  duration: "auto" | "15-30" | "30-60" | "60-90";
  platform: "universal" | "shorts" | "reels" | "tiktok";
  aspect: "9:16";
  style: "auto" | "dynamic" | "editorial" | "podcast" | "gaming";
  framing: FramingMode;
  min_score: number | null;
}

export interface Stage {
  key: string;
  label: string;
  status: "pending" | "active" | "done" | "skipped";
  progress: number;
}

export interface Dimension {
  key: string;
  label: string;
  value: number;
  weight: number;
}

export interface HookType {
  type: string;
  label: string;
  strength: number;
}

export interface CaptionPage {
  start: number;
  end: number;
  text: string;
  k0: number;
  k1: number;
}

export interface Clip {
  id: string;
  rank: number;
  status: "pending" | "queued" | "rendering" | "ready" | "failed";
  progress: number;
  start: number;
  end: number;
  duration: number;
  score: number;
  score_stale: boolean;
  engine: "heuristic" | "llm+signals" | "signals-only";
  dimensions: Dimension[];
  explanation: string;
  hook: { text?: string; types?: HookType[]; score?: number; opener?: string };
  titles: string[];
  caption: string;
  copy_source: "extractive" | "llm" | "none";
  language: string;
  captions: Partial<CaptionOptions>;
  framing_mode: FramingMode;
  smart_trim: boolean;
  layouts: string[];
  framing_notes: string[];
  has_captions_burned: boolean;
  has_srt: boolean;
  qa: {
    passed: boolean;
    checks: Record<string, boolean>;
    width: number;
    height: number;
    fps: number;
    size_bytes: number;
  } | null;
  version: number;
  error: string;
  urls: { video: string; download: string; srt: string; thumbnail: string } | null;
  // detail only
  transcript?: string;
  pages?: CaptionPage[];
  source_duration?: number;
}

export interface Analysis {
  engine?: Clip["engine"];
  mode?: "speech" | "signals";
  language?: string;
  transcription?: string;
  words?: number;
  candidates?: number;
  requested?: number;
  selected?: number;
  min_score?: number;
  best_candidate_score?: number;
  llm?: { used: boolean; model?: string; reason?: string };
  warnings?: string[];
}

export interface Job {
  id: string;
  status: "uploaded" | "queued" | "processing" | "completed" | "failed" | "canceled";
  stage: string;
  progress: number;
  stages: Stage[];
  error: string;
  filename: string;
  size_bytes: number;
  media: { duration?: number; width?: number; height?: number; fps?: string; has_audio?: boolean };
  options: Partial<JobOptions>;
  analysis: Analysis;
  cancel_requested: boolean;
  created_at: string;
  expires_at: string | null;
  clips: Clip[];
}

export interface LogEntry {
  id: number;
  ts: string;
  level: "info" | "warn" | "error";
  stage: string;
  message: string;
}

export interface AppConfig {
  limits: { max_upload_mb: number; max_duration_s: number; formats: string[]; retention_hours: number };
  caption_styles: { key: CaptionStyle; label: string; description: string; accent: string }[];
  caption_fonts: { key: CaptionOptions["font"]; label: string }[];
  engine: { transcription: string; narrative_analysis: string; llm_enabled: boolean; min_score: number };
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_URL}${path}`, { cache: "no-store", ...init });
  } catch {
    throw new ApiError("Can't reach the processing server. Check that it is running, then try again.", 0);
  }
  if (!res.ok) {
    let detail = `Request failed (${res.status}).`;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = "Some settings are not valid. Review them and try again.";
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail, res.status);
  }
  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const json = (body: unknown): RequestInit => ({
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  config: () => request<AppConfig>("/api/config"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  start: (id: string, options: JobOptions) => request<Job>(`/api/jobs/${id}/start`, { method: "POST", ...json(options) }),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  remove: (id: string) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),
  logs: (id: string, after = 0) => request<{ logs: LogEntry[] }>(`/api/jobs/${id}/logs?after=${after}`),
  clip: (jobId: string, clipId: string) => request<Clip>(`/api/jobs/${jobId}/clips/${clipId}`),
  updateClip: (jobId: string, clipId: string, body: ClipUpdate) =>
    request<Clip>(`/api/jobs/${jobId}/clips/${clipId}`, { method: "PATCH", ...json(body) }),
};

export interface ClipUpdate {
  start?: number;
  end?: number;
  captions?: CaptionOptions;
  framing_mode?: FramingMode;
  smart_trim?: boolean;
  page_edits?: { index: number; text: string }[];
}

/** Absolute URL for a media path returned by the API. */
export const mediaUrl = (path: string) => `${API_URL}${path}`;
export const exportUrl = (jobId: string) => `${API_URL}/api/jobs/${jobId}/export.zip`;

export interface UploadHandle {
  promise: Promise<Job>;
  abort: () => void;
}

/** Upload with real byte-level progress (fetch cannot report upload progress). */
export function uploadVideo(file: File, onProgress: (fraction: number) => void): UploadHandle {
  const xhr = new XMLHttpRequest();
  const promise = new Promise<Job>((resolve, reject) => {
    xhr.open("POST", `${API_URL}/api/jobs?filename=${encodeURIComponent(file.name)}`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(e.loaded / e.total);
    };
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        onProgress(1);
        resolve(xhr.response as Job);
      } else {
        const detail = (xhr.response as { detail?: unknown } | null)?.detail;
        reject(new ApiError(typeof detail === "string" ? detail : `Upload failed (${xhr.status}).`, xhr.status));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError("The upload was interrupted. Check your connection and that the server is running.", 0));
    xhr.onabort = () => reject(new ApiError("Upload canceled.", -1));
    xhr.send(file);
  });
  return { promise, abort: () => xhr.abort() };
}

export const DEFAULT_CAPTIONS: CaptionOptions = {
  enabled: true,
  style: "high_retention",
  highlight_keywords: true,
  size: "m",
  position: "bottom",
  color: "",
  font: "auto",
  animation: true,
};

export const DEFAULT_OPTIONS: JobOptions = {
  captions: DEFAULT_CAPTIONS,
  language: "auto",
  clip_count: "auto",
  duration: "auto",
  platform: "universal",
  aspect: "9:16",
  style: "auto",
  framing: "auto",
  min_score: null,
};

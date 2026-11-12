/** Typed client for the ViralCut API. The browser talks to the API directly. */

export const API_URL = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");

export type CaptionStyle =
  | "auto"
  | "viral_dynamic"
  | "energy"
  | "podcast"
  | "kinetic"
  | "clean_editorial"
  | "gaming"
  | "cinematic"
  | "high_retention"
  | "dynamic_creator";
export type FramingMode = "auto" | "face" | "speaker" | "split" | "blur" | "center";
export type VideoStyle =
  | "auto"
  | "viral"
  | "podcast"
  | "gaming"
  | "cinematic"
  | "educational"
  | "reaction"
  | "minimal"
  | "storytime"
  | "news";
export type Intensity = "subtle" | "balanced" | "high";
export type ContentType = "shorts" | "tiktok" | "reels" | "youtube" | "compilation";
/** Creator Studio: per-editor settings keyed by the editor's command name (corte, 3d, tela-verde…). */
export type EditorsConfig = Record<string, Record<string, unknown>>;

export interface EditorControl {
  name: string;
  type: "number" | "bool" | "select" | "text" | "color" | "asset" | "proofs";
  label: string;
  min?: number;
  max?: number;
  step?: number;
  options?: string[];
  kind?: "music" | "image" | "reference";
  auto?: string;
}

export interface EditorInfo {
  key: string;
  command: string;
  label: string;
  description: string;
  heavy: boolean;
  controls: EditorControl[];
}

export interface PresetInfo {
  key: string;
  label: string;
  description: string;
  options: Partial<JobOptions> & { captions?: Partial<CaptionOptions> };
  editors: EditorsConfig;
}

export interface Asset {
  id: string;
  kind: "music" | "image" | "reference";
  name: string;
  size: number;
  duration?: number;
  width?: number;
  height?: number;
}

export interface Growth {
  titles: string[];
  description: string;
  keywords: string[];
  hook: string;
  thumbnail_text: string[];
  category: string;
  hashtags: string[];
  platforms: Record<string, string>;
  followups: string[];
  versions_to_test: string[];
  chapters?: { t: number; title: string }[];
  note: string;
}

export interface JobSummary {
  id: string;
  status: Job["status"];
  filename: string;
  created_at: string;
  media: Job["media"];
  options: Partial<JobOptions>;
  clip_count: number;
  thumbnail: string | null;
}

export interface CaptionOptions {
  enabled: boolean;
  style: CaptionStyle;
  highlight_keywords: boolean;
  size: "s" | "m" | "l" | "xl";
  position: "top" | "middle" | "bottom";
  color: string;
  font: "auto" | "inter" | "poppins" | "anton" | "dm_serif";
  animation: boolean;
  // Caption Studio overrides (empty / "auto" / null = the style's own value)
  text_color?: string;
  keyword_color?: string;
  weight?: "auto" | "medium" | "bold" | "heavy";
  case?: "auto" | "upper" | "normal";
  words_per_page?: number | null;
  max_lines?: number | null;
  line_spacing?: number | null;
  outline?: number | null;
  shadow?: number | null;
  background?: "auto" | "none" | "box";
  animation_type?: "auto" | "color" | "pop" | "box" | "reveal" | "none";
  animation_intensity?: "subtle" | "normal" | "strong";
  keyword_mode?: "auto" | "color" | "color_scale" | "off";
  offset?: number;
}

export interface JobOptions {
  captions: CaptionOptions;
  language: string;
  clip_count: number | "auto";
  duration: "auto" | "15-30" | "30-60" | "60-90" | "custom";
  custom_min: number;
  custom_max: number;
  platform: "universal" | "shorts" | "reels" | "tiktok" | "youtube";
  aspect: "9:16" | "16:9";
  style: VideoStyle;
  intensity: Intensity;
  variations: number;
  quality: "high" | "max";
  performance: "auto" | "eco" | "balanced" | "turbo";
  content_type: ContentType;
  preset: string;
  editors: EditorsConfig;
  growth: boolean;
  framing: FramingMode;
  smart_zoom: boolean;
  review_transcript: boolean;
  glossary: string;
  min_score: number | null;
}

export interface Stage {
  key: string;
  label: string;
  status: "pending" | "active" | "done" | "skipped";
  progress: number;
  elapsed_s?: number | null;
}

export interface StyleRecommendation {
  style: VideoStyle;
  label: string;
  intensity: Intensity;
  reasons: string[];
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
  style: VideoStyle | "dynamic" | "editorial";
  intensity: Intensity;
  quality: "high" | "max";
  applied_style: { style: VideoStyle; label: string; chosen_by: "user" | "auto"; reason: string; intensity: Intensity } | null;
  variant: { index: number; kind: string; label: string; why: string; of: string } | null;
  recommendation: StyleRecommendation | null;
  kind: "clip" | "compilation";
  aspect: "9:16" | "16:9";
  growth: Growth | null;
  hook_first: { score: number; from: number; text: string; skipped_s: number } | null;
  editors: EditorsConfig;
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
  asr?: { mode?: string; model?: string; dropped_segments?: { start: number; end: number; text: string; reason: string }[]; recovered_gaps?: { start: number; end: number; words: number }[] };
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
  status: "uploaded" | "queued" | "processing" | "review" | "completed" | "failed" | "canceled";
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

export interface PerformanceInfo {
  profiles: { key: "eco" | "balanced" | "turbo"; threads: number; render_workers: number; nice: number; keep_asr_model: boolean }[];
  default: "eco" | "balanced" | "turbo";
  recommended: "eco" | "balanced" | "turbo";
  reason: string;
  hardware: { logical_cpus: number; physical_cpus: number; ram_gb: number | null; gpu: { name?: string; vram_gb?: number } };
}

export interface AppConfig {
  limits: { max_upload_mb: number; max_duration_s: number; formats: string[]; retention_hours: number };
  caption_styles: { key: CaptionStyle; label: string; description: string; accent: string }[];
  editors: EditorInfo[];
  presets: PresetInfo[];
  content_types: ContentType[];
  video_styles: { key: VideoStyle; label: string; description: string; caption: CaptionStyle; framing: FramingMode; best_for: string[] }[];
  performance: PerformanceInfo;
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

export interface Health {
  status: string;
  version: string;
  /** the worker inside the API process; "off" means jobs are run by a separate worker process */
  worker?: { state: "off" | "starting" | "idle" | "busy" | "error"; error: string; job: string };
}

export const api = {
  config: () => request<AppConfig>("/api/config"),
  health: () => request<Health>("/api/health"),
  job: (id: string) => request<Job>(`/api/jobs/${id}`),
  start: (id: string, options: JobOptions) => request<Job>(`/api/jobs/${id}/start`, { method: "POST", ...json(options) }),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  remove: (id: string) => request<void>(`/api/jobs/${id}`, { method: "DELETE" }),
  logs: (id: string, after = 0) => request<{ logs: LogEntry[] }>(`/api/jobs/${id}/logs?after=${after}`),
  clip: (jobId: string, clipId: string) => request<Clip>(`/api/jobs/${jobId}/clips/${clipId}`),
  transcript: (id: string) => request<Transcript>(`/api/jobs/${id}/transcript`),
  saveTranscript: (id: string, body: TranscriptUpdate) =>
    request<{ changed: number; affected_clips: string[]; words: number }>(`/api/jobs/${id}/transcript`, { method: "PUT", ...json(body) }),
  resume: (id: string) => request<Job>(`/api/jobs/${id}/resume`, { method: "POST" }),
  updateClip: (jobId: string, clipId: string, body: ClipUpdate) =>
    request<Clip>(`/api/jobs/${jobId}/clips/${clipId}`, { method: "PATCH", ...json(body) }),
  jobs: () => request<{ jobs: JobSummary[] }>("/api/jobs"),
  assets: (jobId: string) => request<{ assets: Asset[] }>(`/api/jobs/${jobId}/assets`),
  uploadAsset: async (jobId: string, kind: Asset["kind"], file: File): Promise<Asset> => {
    const res = await fetch(`${API_URL}/api/jobs/${jobId}/assets?kind=${kind}&filename=${encodeURIComponent(file.name)}`, { method: "POST", body: file });
    if (!res.ok) {
      let detail = `Upload failed (${res.status}).`;
      try {
        const b = (await res.json()) as { detail?: unknown };
        if (typeof b.detail === "string") detail = b.detail;
      } catch {
        /* non-JSON */
      }
      throw new ApiError(detail, res.status);
    }
    return (await res.json()) as Asset;
  },
  /** JPEG rendered by the export engine itself (same layout code, same libass). */
  captionPreview: async (body: CaptionPreviewRequest, signal?: AbortSignal): Promise<Blob> => {
    const res = await fetch(`${API_URL}/api/caption-preview`, { method: "POST", signal, ...json(body) });
    if (!res.ok) throw new ApiError(res.status === 429 ? "Preview busy" : "Preview failed", res.status);
    return res.blob();
  },
};

export interface CaptionPreviewRequest {
  captions: CaptionOptions;
  platform?: JobOptions["platform"];
  text?: string;
  language?: string;
  job_id?: string;
  clip_id?: string;
}

export interface TranscriptWord {
  index: number;
  text: string;
  start: number;
  end: number;
  confidence: number;
}

export interface Transcript {
  language: string;
  engine: string;
  words: TranscriptWord[];
  quality: { dropped_segments?: { start: number; end: number; text: string; reason: string }[] };
  edited: boolean;
}

export interface TranscriptUpdate {
  words?: { index: number; text: string }[];
  replace?: { find: string; replace: string; case_sensitive?: boolean }[];
}

export interface ClipUpdate {
  start?: number;
  end?: number;
  captions?: CaptionOptions;
  framing_mode?: FramingMode;
  smart_trim?: boolean;
  style?: VideoStyle;
  intensity?: Intensity;
  quality?: "high" | "max";
  editors?: EditorsConfig;
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
  custom_min: 20,
  custom_max: 45,
  platform: "universal",
  aspect: "9:16",
  style: "auto",
  intensity: "balanced",
  variations: 1,
  quality: "high",
  performance: "auto",
  content_type: "shorts",
  preset: "",
  editors: {},
  growth: true,
  framing: "auto",
  smart_zoom: true,
  review_transcript: false,
  glossary: "",
  min_score: null,
};

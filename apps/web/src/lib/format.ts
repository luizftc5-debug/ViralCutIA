/** 83.4 -> "1:23", 3723 -> "1:02:03" */
export function timecode(seconds: number | undefined | null, withTenths = false): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  const s = Math.max(0, seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = withTenths ? (s % 60).toFixed(1).padStart(4, "0") : String(Math.floor(s % 60)).padStart(2, "0");
  return h > 0 ? `${h}:${String(m).padStart(2, "0")}:${sec}` : `${m}:${sec}`;
}

export function fileSize(bytes: number | undefined | null): string {
  if (!bytes) return "–";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i++;
  }
  return `${v >= 100 || i === 0 ? v.toFixed(0) : v.toFixed(1)} ${units[i]}`;
}

/** Elapsed time in words: 8 -> "8 s", 125 -> "2 min 05 s", 3900 -> "1 h 05 min" */
export function elapsed(value: number | undefined | null): string {
  if (value == null || !Number.isFinite(value)) return "";
  const s = Math.max(0, Math.round(value));
  if (s < 60) return `${s} s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m} min ${String(s % 60).padStart(2, "0")} s`;
  return `${Math.floor(m / 60)} h ${String(m % 60).padStart(2, "0")} min`;
}

export function seconds(value: number): string {
  return value >= 60 ? timecode(value) : `${Math.round(value)}s`;
}

const LANGUAGES: Record<string, string> = {
  en: "English", pt: "Portuguese", es: "Spanish", fr: "French", de: "German", it: "Italian",
  ja: "Japanese", ko: "Korean", zh: "Chinese", hi: "Hindi", ar: "Arabic", ru: "Russian",
  nl: "Dutch", tr: "Turkish", pl: "Polish",
};

export function languageName(code: string | undefined): string {
  if (!code) return "";
  return LANGUAGES[code.slice(0, 2)] ?? code.toUpperCase();
}

export const LANGUAGE_OPTIONS = [{ value: "auto", label: "Detect automatically" }].concat(
  Object.entries(LANGUAGES).map(([value, label]) => ({ value, label })),
);

const LAYOUTS: Record<string, string> = {
  crop: "Subject tracking",
  split: "Split screen",
  blur: "Blurred background",
  fit: "Original framing",
};

export const layoutName = (key: string) => LAYOUTS[key] ?? key;

/** Deterministic pseudo-random in [0,1) — used for illustrative waveforms so SSR and client agree. */
export function seeded(i: number, salt = 1): number {
  const x = Math.sin(i * 127.1 + salt * 311.7) * 43758.5453;
  return x - Math.floor(x);
}

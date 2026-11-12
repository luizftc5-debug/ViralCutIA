"use client";

import { Upload } from "lucide-react";
import { useRouter } from "next/navigation";
import { useId, useRef, useState } from "react";

import { seeded } from "@/lib/format";

import { cx } from "./ui";
import { useUpload } from "./upload-provider";

const BARS = 132;
// Illustrative cut ranges (as fractions of the strip) — a picture of the idea, not data.
const PICKS = [
  { from: 0.06, to: 0.17 },
  { from: 0.72, to: 0.8 },
  { from: 0.87, to: 0.95 },
];

function limitsLine(formats: string[], maxMb?: number, maxS?: number): string {
  const list = formats.length > 1 ? `${formats.slice(0, -1).join(", ")} or ${formats.at(-1)}` : (formats[0] ?? "");
  const size = maxMb ? `, up to ${maxMb >= 1024 ? `${Math.round(maxMb / 1024)} GB` : `${maxMb} MB`}` : "";
  const length = maxS ? ` and ${Math.round(maxS / 3600)} hours long` : "";
  return `${list}${size}${length}`;
}

/**
 * The upload target, drawn as an empty editing timeline. Dropping a file starts the upload
 * immediately and moves to the studio, where settings are chosen while bytes are in flight.
 */
export function TimelineDropzone({ compact = false }: { compact?: boolean }) {
  const { begin, config } = useUpload();
  const router = useRouter();
  const inputRef = useRef<HTMLInputElement>(null);
  const inputId = useId();
  const [over, setOver] = useState(false);
  const [error, setError] = useState("");

  const accept = (file: File | undefined) => {
    if (!file) return;
    const problem = begin(file);
    if (problem) {
      setError(problem);
      return;
    }
    setError("");
    router.push("/new");
  };

  const formats = config?.limits.formats ?? ["MP4", "MOV", "MKV", "WEBM"];

  return (
    <div>
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault();
          setOver(false);
          accept(e.dataTransfer.files[0]);
        }}
        className={cx(
          "relative overflow-hidden rounded-(--radius-frame) border bg-panel transition-colors",
          over ? "border-amber" : "border-line-strong",
        )}
      >
        {/* ruler */}
        <div aria-hidden className="flex h-7 items-end border-b border-line px-4 sm:px-6">
          <div className="flex w-full justify-between">
            {Array.from({ length: 13 }, (_, i) => (
              <span key={i} className={cx("w-px bg-line-strong", i % 3 === 0 ? "h-3" : "h-1.5")} />
            ))}
          </div>
        </div>

        {/* track */}
        <div className={cx("relative px-4 sm:px-6", compact ? "py-8" : "py-10 sm:py-14")}>
          <div aria-hidden className={cx("pointer-events-none absolute inset-x-4 flex items-center gap-[3px] sm:inset-x-6", compact ? "inset-y-6" : "inset-y-8")}>
            {Array.from({ length: BARS }, (_, i) => {
              const x = i / BARS;
              const picked = PICKS.some((p) => x >= p.from && x <= p.to);
              const h = 18 + seeded(i, 3) * 52 + (picked ? 22 : 0);
              return (
                <span
                  key={i}
                  className={cx("flex-1 origin-center animate-draw rounded-full", picked ? "bg-amber/45" : "bg-line-strong/70", i % 2 === 1 && "hidden sm:block")}
                  style={{ height: `${Math.min(100, h).toFixed(1)}%`, animationDelay: `${i * 4}ms` }}
                />
              );
            })}
            {PICKS.map((p, i) => (
              <span
                key={i}
                className="absolute inset-y-0 animate-snap border-x-2 border-amber before:absolute before:left-0 before:top-0 before:h-0.5 before:w-2 before:bg-amber after:absolute after:bottom-0 after:left-0 after:h-0.5 after:w-2 after:bg-amber"
                style={{ left: `${p.from * 100}%`, width: `${(p.to - p.from) * 100}%`, animationDelay: `${700 + i * 140}ms` }}
              >
                <span className="absolute right-0 top-0 h-0.5 w-2 bg-amber" />
                <span className="absolute bottom-0 right-0 h-0.5 w-2 bg-amber" />
              </span>
            ))}
          </div>

          <div className="relative mx-auto flex max-w-md flex-col items-center rounded-xl bg-black/80 px-6 py-6 text-center backdrop-blur-sm">
            <label
              htmlFor={inputId}
              className="inline-flex h-12 items-center gap-2.5 rounded-lg bg-fg px-6 text-[15px] font-medium text-black transition-colors hover:bg-white has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-amber"
            >
              <Upload className="size-4" aria-hidden />
              Choose a video
              <input
                id={inputId}
                ref={inputRef}
                type="file"
                className="sr-only"
                accept=".mp4,.mov,.mkv,.webm,video/mp4,video/quicktime,video/x-matroska,video/webm"
                onChange={(e) => {
                  accept(e.target.files?.[0]);
                  e.target.value = "";
                }}
              />
            </label>
            <p className="mt-3 text-sm text-muted">{over ? "Drop it on the timeline" : "or drag it onto this timeline"}</p>
            <p className="tnum mt-1 text-[13px] text-faint">{limitsLine(formats, config?.limits.max_upload_mb, config?.limits.max_duration_s)}</p>
          </div>
        </div>
      </div>
      {error && (
        <p role="alert" className="mt-3 text-sm text-danger">
          {error}
        </p>
      )}
    </div>
  );
}

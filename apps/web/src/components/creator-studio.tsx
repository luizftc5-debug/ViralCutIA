"use client";

import { ImagePlus, Music, Upload, X } from "lucide-react";
import { useEffect, useId, useState } from "react";

import {
  api,
  type AppConfig,
  type Asset,
  type ContentType,
  type EditorControl,
  type EditorInfo,
  type EditorsConfig,
  type JobOptions,
  type PresetInfo,
} from "@/lib/api";

import { cx, Notice, Toggle } from "./ui";

/** Mirrors pipeline/editors.py apply_preset: a preset sets real options and switches editors on. */
export function applyPreset(
  options: JobOptions,
  preset: PresetInfo,
): JobOptions {
  const next: JobOptions = { ...options, preset: preset.key };
  for (const [k, v] of Object.entries(preset.options)) {
    if (k === "clip_count" && options.clip_count !== "auto") continue; // the user's own number wins
    if (k === "captions")
      next.captions = { ...options.captions, ...(v as object) };
    else (next as unknown as Record<string, unknown>)[k] = v;
  }
  const editors: EditorsConfig = {};
  for (const [k, cfg] of Object.entries(options.editors))
    editors[k] = { ...cfg };
  for (const [k, cfg] of Object.entries(preset.editors))
    editors[k] = { ...(editors[k] ?? {}), ...cfg };
  next.editors = editors;
  return next;
}

const CONTENT_TYPES: { key: ContentType; label: string; body: string }[] = [
  {
    key: "shorts",
    label: "YouTube Shorts",
    body: "9:16, strong opening, concise edit",
  },
  {
    key: "tiktok",
    label: "TikTok",
    body: "9:16, mobile-first, TikTok safe areas",
  },
  {
    key: "reels",
    label: "Instagram Reels",
    body: "9:16, Reels safe text placement",
  },
  {
    key: "youtube",
    label: "YouTube long-form",
    body: "16:9, minutes long, chapters",
  },
  {
    key: "compilation",
    label: "Compilation / ranking",
    body: "Best moments counted down in one video",
  },
];

export function ContentTypePicker({
  value,
  onChange,
}: {
  value: ContentType;
  onChange: (v: ContentType) => void;
}) {
  const name = useId();
  return (
    <fieldset>
      <legend className="mb-2 text-sm text-muted">Content type</legend>
      <div className="grid gap-2 sm:grid-cols-3 lg:grid-cols-5">
        {CONTENT_TYPES.map((c) => (
          <label
            key={c.key}
            className={cx(
              "block rounded-xl border bg-panel px-3 py-2.5 has-[:focus-visible]:outline has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-amber",
              value === c.key
                ? "border-amber"
                : "border-line hover:border-line-strong",
            )}
          >
            <input
              type="radio"
              name={name}
              className="sr-only"
              checked={value === c.key}
              onChange={() => onChange(c.key)}
            />
            <span className="block text-sm font-medium">{c.label}</span>
            <span className="mt-0.5 block text-[12px] leading-snug text-faint">
              {c.body}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}

export function PresetGrid({
  presets,
  value,
  onPick,
}: {
  presets: PresetInfo[];
  value: string;
  onPick: (p: PresetInfo) => void;
}) {
  return (
    <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
      {presets.map((p) => (
        <button
          key={p.key}
          type="button"
          onClick={() => onPick(p)}
          aria-pressed={value === p.key}
          className={cx(
            "rounded-xl border bg-panel px-3 py-3 text-left transition-colors",
            value === p.key
              ? "border-amber"
              : "border-line hover:border-line-strong",
          )}
        >
          <span className="block text-sm font-medium">{p.label}</span>
          <span className="mt-0.5 block text-[12px] leading-snug text-faint">
            {p.description}
          </span>
          {Object.keys(p.editors).length > 0 && (
            <span className="mt-1.5 block text-[11px] text-muted">
              Editors:{" "}
              {Object.keys(p.editors)
                .map((k) => `/${k}`)
                .join(" ")}
            </span>
          )}
        </button>
      ))}
    </div>
  );
}

function AssetPicker({
  jobId,
  kind,
  value,
  assets,
  onUploaded,
  onChange,
}: {
  jobId?: string;
  kind: Asset["kind"];
  value: string;
  assets: Asset[];
  onUploaded: (a: Asset) => void;
  onChange: (id: string) => void;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");
  const mine = assets.filter((a) => a.kind === kind);
  const accept =
    kind === "music"
      ? "audio/*"
      : kind === "image"
        ? "image/png,image/jpeg,image/webp"
        : "video/*";
  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2">
        <select
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="h-9 rounded-md border border-line bg-panel px-2 text-sm text-fg"
          aria-label={`Choose ${kind}`}
        >
          <option value="">None</option>
          {mine.map((a) => (
            <option key={a.id} value={a.id}>
              {a.name}
            </option>
          ))}
        </select>
        <label
          className={cx(
            "inline-flex h-9 cursor-pointer items-center gap-1.5 rounded-md border border-line px-3 text-[13px] text-muted hover:text-fg",
            !jobId && "pointer-events-none opacity-50",
          )}
        >
          {kind === "music" ? (
            <Music className="size-3.5" aria-hidden />
          ) : kind === "image" ? (
            <ImagePlus className="size-3.5" aria-hidden />
          ) : (
            <Upload className="size-3.5" aria-hidden />
          )}
          {busy ? "Uploading…" : "Upload"}
          <input
            type="file"
            accept={accept}
            className="sr-only"
            disabled={!jobId || busy}
            onChange={async (e) => {
              const f = e.target.files?.[0];
              e.target.value = "";
              if (!f || !jobId) return;
              setBusy(true);
              setErr("");
              try {
                const a = await api.uploadAsset(jobId, kind, f);
                onUploaded(a);
                onChange(a.id);
              } catch (x) {
                setErr(x instanceof Error ? x.message : "Upload failed");
              } finally {
                setBusy(false);
              }
            }}
          />
        </label>
      </div>
      {!jobId && (
        <p className="text-[12px] text-faint">
          Available once your video has uploaded.
        </p>
      )}
      {err && <p className="text-[12px] text-red-400">{err}</p>}
    </div>
  );
}

function Control({
  c,
  value,
  set,
  jobId,
  assets,
  addAsset,
}: {
  c: EditorControl;
  value: unknown;
  set: (v: unknown) => void;
  jobId?: string;
  assets: Asset[];
  addAsset: (a: Asset) => void;
}) {
  if (c.type === "bool")
    return <Toggle label={c.label} checked={Boolean(value)} onChange={set} />;
  if (c.type === "select")
    return (
      <label className="block text-sm text-muted">
        {c.label}
        <select
          value={String(value ?? c.options?.[0] ?? "")}
          onChange={(e) => set(e.target.value)}
          className="mt-1 block h-9 w-full rounded-md border border-line bg-panel px-2 text-fg"
        >
          {c.options?.map((o) => (
            <option key={o} value={o}>
              {o}
            </option>
          ))}
        </select>
      </label>
    );
  if (c.type === "number")
    return (
      <label className="block text-sm text-muted">
        <span className="flex justify-between">
          {c.label}
          <span className="tnum text-fg">
            {value === null || value === undefined
              ? (c.auto ?? "auto")
              : String(value)}
          </span>
        </span>
        <input
          type="range"
          min={c.min}
          max={c.max}
          step={c.step}
          value={Number(value ?? c.min ?? 0)}
          onChange={(e) => set(Number(e.target.value))}
          className="mt-1 w-full"
        />
      </label>
    );
  if (c.type === "text")
    return (
      <label className="block text-sm text-muted">
        {c.label}
        <input
          type="text"
          maxLength={80}
          value={String(value ?? "")}
          onChange={(e) => set(e.target.value)}
          className="mt-1 block h-9 w-full rounded-md border border-line bg-panel px-2 text-fg"
        />
      </label>
    );
  if (c.type === "color")
    return (
      <label className="flex items-center gap-2 text-sm text-muted">
        {c.label}
        <input
          type="color"
          value={String(value || "#FFD60A")}
          onChange={(e) => set(e.target.value.toUpperCase())}
          className="h-8 w-10 rounded border border-line bg-panel"
        />
        {Boolean(value) && (
          <button
            type="button"
            className="text-[12px] underline"
            onClick={() => set("")}
          >
            reset
          </button>
        )}
      </label>
    );
  if (c.type === "asset" && c.kind)
    return (
      <div className="text-sm text-muted">
        <span className="mb-1 block">{c.label}</span>
        <AssetPicker
          jobId={jobId}
          kind={c.kind}
          value={String(value ?? "")}
          assets={assets}
          onUploaded={addAsset}
          onChange={(id) => set(id)}
        />
      </div>
    );
  if (c.type === "proofs") {
    const items =
      (value as
        | {
            asset: string;
            at: number | null;
            keyword: string;
            duration: number;
          }[]
        | undefined) ?? [];
    return (
      <div className="space-y-2 text-sm text-muted">
        <span className="block">{c.label}</span>
        {items.map((it, i) => (
          <div
            key={i}
            className="flex flex-wrap items-center gap-2 rounded-md border border-line p-2"
          >
            <span className="max-w-40 truncate text-fg">
              {assets.find((a) => a.id === it.asset)?.name ?? "image"}
            </span>
            <input
              type="text"
              placeholder="show when this word is said"
              value={it.keyword}
              onChange={(e) =>
                set(
                  items.map((x, j) =>
                    j === i ? { ...x, keyword: e.target.value } : x,
                  ),
                )
              }
              className="h-8 flex-1 rounded border border-line bg-panel px-2 text-fg"
            />
            <input
              type="number"
              min={0}
              step={0.5}
              placeholder="or at second"
              value={it.at ?? ""}
              onChange={(e) =>
                set(
                  items.map((x, j) =>
                    j === i
                      ? {
                          ...x,
                          at:
                            e.target.value === ""
                              ? null
                              : Number(e.target.value),
                        }
                      : x,
                  ),
                )
              }
              className="h-8 w-28 rounded border border-line bg-panel px-2 text-fg"
              aria-label="Time in seconds"
            />
            <button
              type="button"
              aria-label="Remove"
              onClick={() => set(items.filter((_, j) => j !== i))}
              className="p-1 hover:text-fg"
            >
              <X className="size-4" aria-hidden />
            </button>
          </div>
        ))}
        <AssetPicker
          jobId={jobId}
          kind="image"
          value=""
          assets={assets}
          onUploaded={addAsset}
          onChange={(id) =>
            id &&
            set([...items, { asset: id, at: null, keyword: "", duration: 3 }])
          }
        />
      </div>
    );
  }
  return null;
}

const DEFAULT_ON = new Set([
  "corte",
  "tomada",
  "ganchos",
  "legenda",
  "enquadramento",
]);

export function isEditorOn(editors: EditorsConfig, key: string): boolean {
  const v = editors[key]?.enabled;
  return v === undefined ? DEFAULT_ON.has(key) : Boolean(v);
}

/** The 13 specialised editors, each with its own switch and real settings (see pipeline/editors.py). */
export function EditorsPanel({
  editors,
  onChange,
  info,
  jobId,
  captionsOn,
  onCaptions,
}: {
  editors: EditorsConfig;
  onChange: (e: EditorsConfig) => void;
  info?: AppConfig["editors"];
  jobId?: string;
  captionsOn: boolean;
  onCaptions: (v: boolean) => void;
}) {
  const [assets, setAssets] = useState<Asset[]>([]);
  useEffect(() => {
    if (!jobId) return;
    let alive = true;
    api
      .assets(jobId)
      .then((r) => alive && setAssets(r.assets))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [jobId]);
  if (!info) return <p className="text-sm text-faint">Loading editors…</p>;
  const setField = (key: string, field: string, v: unknown) =>
    onChange({ ...editors, [key]: { ...(editors[key] ?? {}), [field]: v } });
  const anyHeavy = info.some((e) => e.heavy && isEditorOn(editors, e.key));
  return (
    <div className="space-y-3">
      {anyHeavy && (
        <Notice tone="warn">
          Heavy editors are on (3D camera, AI background, text behind subject).
          They add render time; Eco mode keeps the computer usable while they
          run.
        </Notice>
      )}
      <div className="grid gap-3 lg:grid-cols-2">
        {info.map((e: EditorInfo) => {
          const on =
            e.key === "legenda" ? captionsOn : isEditorOn(editors, e.key);
          return (
            <div
              key={e.key}
              className={cx(
                "rounded-xl border bg-panel p-4",
                on ? "border-amber/60" : "border-line",
              )}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">
                    <span className="font-mono text-amber">{e.command}</span>{" "}
                    {e.label}
                    {e.heavy && (
                      <span className="ml-2 rounded border border-line px-1.5 py-0.5 text-[11px] text-faint">
                        heavy
                      </span>
                    )}
                  </p>
                  <p className="mt-1 text-[13px] leading-snug text-faint">
                    {e.description}
                  </p>
                </div>
                <label className="relative inline-flex shrink-0 cursor-pointer items-center">
                  <input
                    type="checkbox"
                    className="peer sr-only"
                    checked={on}
                    onChange={(ev) =>
                      e.key === "legenda"
                        ? onCaptions(ev.target.checked)
                        : setField(e.key, "enabled", ev.target.checked)
                    }
                    aria-label={`${e.command} ${on ? "on" : "off"}`}
                  />
                  <span className="h-6 w-10 rounded-full bg-line transition-colors peer-checked:bg-amber" />
                  <span className="absolute left-0.5 top-0.5 size-5 rounded-full bg-fg transition-transform peer-checked:translate-x-4" />
                </label>
              </div>
              {on && e.controls.length > 0 && (
                <div className="mt-4 space-y-3 border-t border-line pt-3">
                  {e.controls.map((c) => (
                    <Control
                      key={c.name}
                      c={c}
                      value={editors[e.key]?.[c.name]}
                      set={(v) => setField(e.key, c.name, v)}
                      jobId={jobId}
                      assets={assets}
                      addAsset={(a) => setAssets((xs) => [...xs, a])}
                    />
                  ))}
                </div>
              )}
              {on && e.key === "legenda" && (
                <p className="mt-3 text-[12px] text-faint">
                  Caption styles and Caption Studio are in the Captions section.
                </p>
              )}
              {on && e.key === "enquadramento" && (
                <p className="mt-3 text-[12px] text-faint">
                  Visual format and punch-ins are set in the Clips section.
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

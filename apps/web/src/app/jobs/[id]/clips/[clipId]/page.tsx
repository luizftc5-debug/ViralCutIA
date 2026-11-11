"use client";

import { ArrowLeft, Download, Minus, Plus } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { CaptionControls } from "@/components/caption-controls";
import { ScoreNumber } from "@/components/score";
import { SiteHeader } from "@/components/site-header";
import { Button, buttonClass, Notice, ProgressBar, Segmented, Toggle } from "@/components/ui";
import { api, ApiError, type CaptionOptions, type Clip, type ClipUpdate, DEFAULT_CAPTIONS, type FramingMode, mediaUrl } from "@/lib/api";
import { layoutName, timecode } from "@/lib/format";

interface Draft {
  start: number;
  end: number;
  captions: CaptionOptions;
  framing: FramingMode;
  smartTrim: boolean;
  pages: Record<number, string>;
}

function draftFrom(clip: Clip): Draft {
  return {
    start: clip.start,
    end: clip.end,
    captions: { ...DEFAULT_CAPTIONS, ...clip.captions },
    framing: clip.framing_mode,
    smartTrim: clip.smart_trim,
    pages: {},
  };
}

function TimeField({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  const clamp = (v: number) => Math.round(Math.max(min, Math.min(max, v)) * 10) / 10;
  return (
    <div>
      <label className="mb-2 block text-sm text-muted">
        {label} <span className="tnum text-faint">({timecode(value, true)} in the source)</span>
      </label>
      <div className="inline-flex items-center rounded-lg border border-line bg-panel">
        <button type="button" aria-label={`${label} half a second earlier`} onClick={() => onChange(clamp(value - 0.5))} className="flex size-10 items-center justify-center text-muted hover:text-fg">
          <Minus className="size-4" aria-hidden />
        </button>
        <input
          type="number"
          step="any"
          min={min}
          max={max}
          value={value}
          onChange={(e) => onChange(clamp(Number(e.target.value)))}
          aria-label={`${label}, in seconds from the start of the source video`}
          className="tnum h-10 w-24 border-x border-line bg-transparent text-center text-fg [appearance:textfield]"
        />
        <button type="button" aria-label={`${label} half a second later`} onClick={() => onChange(clamp(value + 0.5))} className="flex size-10 items-center justify-center text-muted hover:text-fg">
          <Plus className="size-4" aria-hidden />
        </button>
      </div>
    </div>
  );
}

export default function ClipStudioPage() {
  const { id, clipId } = useParams<{ id: string; clipId: string }>();
  const [clip, setClip] = useState<Clip | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [loadError, setLoadError] = useState("");
  const [saveError, setSaveError] = useState("");
  const [saving, setSaving] = useState(false);
  const loadedVersion = useRef(-1);
  const videoRef = useRef<HTMLVideoElement>(null);

  const [nonce, setNonce] = useState(0);

  // Load the clip, and keep polling while a render is queued or running.
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const tick = async () => {
      try {
        const next = await api.clip(id, clipId);
        if (!alive) return;
        setClip(next);
        setLoadError("");
        const busy = next.status === "queued" || next.status === "rendering";
        // (re)seed the form whenever a new rendered version arrives; a failed render keeps your edits
        if (!busy && next.version !== loadedVersion.current) {
          loadedVersion.current = next.version;
          setDraft(draftFrom(next));
        }
        if (busy) timer = setTimeout(tick, 1000);
      } catch (e) {
        if (!alive) return;
        setLoadError(
          e instanceof ApiError && e.status === 404
            ? "This clip no longer exists. Its files may have been deleted."
            : e instanceof Error
              ? e.message
              : "Could not load this clip.",
        );
      }
    };
    void tick();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [id, clipId, nonce]);

  const rendering = clip?.status === "queued" || clip?.status === "rendering";

  const sourceDuration = clip?.source_duration ?? clip?.end ?? 0;

  const changes = useMemo<ClipUpdate | null>(() => {
    if (!clip || !draft) return null;
    const body: ClipUpdate = {};
    if (draft.start !== clip.start) body.start = draft.start;
    if (draft.end !== clip.end) body.end = draft.end;
    const original = { ...DEFAULT_CAPTIONS, ...clip.captions };
    if (JSON.stringify(original) !== JSON.stringify(draft.captions)) body.captions = draft.captions;
    if (draft.framing !== clip.framing_mode) body.framing_mode = draft.framing;
    if (draft.smartTrim !== clip.smart_trim) body.smart_trim = draft.smartTrim;
    const edits = Object.entries(draft.pages)
      .map(([index, text]) => ({ index: Number(index), text }))
      .filter((e) => (clip.pages?.[e.index]?.text ?? "") !== e.text);
    if (edits.length) body.page_edits = edits;
    return Object.keys(body).length ? body : null;
  }, [clip, draft]);

  const lengthProblem = draft && (draft.end - draft.start < 3 ? "A clip needs to be at least 3 seconds long." : draft.end - draft.start > 180 ? "A clip can be at most 3 minutes long." : "");

  const save = async () => {
    if (!changes) return;
    setSaving(true);
    setSaveError("");
    try {
      setClip(await api.updateClip(id, clipId, changes));
      setNonce((n) => n + 1); // start polling the render
    } catch (e) {
      setSaveError(e instanceof Error ? e.message : "Could not start the render.");
    } finally {
      setSaving(false);
    }
  };

  const seekTo = (t: number) => {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = t;
    void v.play().catch(() => undefined);
  };

  if (loadError && !clip) {
    return (
      <>
        <SiteHeader studio />
        <main id="main" className="mx-auto max-w-[1240px] px-5 py-20 sm:px-8">
          <Notice tone="error">{loadError}</Notice>
          <Link href={`/jobs/${id}`} className={`${buttonClass("secondary", "md")} mt-6`}>
            Back to all clips
          </Link>
        </main>
      </>
    );
  }
  if (!clip || !draft) {
    return (
      <>
        <SiteHeader studio />
        <main id="main" className="mx-auto max-w-[1240px] px-5 py-20 text-muted sm:px-8">
          Loading…
        </main>
      </>
    );
  }

  const setD = <K extends keyof Draft>(key: K, v: Draft[K]) => setDraft((d) => (d ? { ...d, [key]: v } : d));
  const trimmed = draft.start !== clip.start || draft.end !== clip.end;

  return (
    <>
      <SiteHeader studio />
      <main id="main" className="mx-auto max-w-[1240px] px-5 pb-32 pt-8 sm:px-8">
        <Link href={`/jobs/${id}`} className="inline-flex items-center gap-1.5 text-sm text-muted hover:text-fg">
          <ArrowLeft className="size-4" aria-hidden />
          All clips
        </Link>

        <div className="mt-6 grid gap-x-14 gap-y-10 lg:grid-cols-[300px_minmax(0,1fr)]">
          {/* preview: always the last rendered file, so what you see is what you download */}
          <aside className="lg:sticky lg:top-24 lg:self-start">
            <div className="relative aspect-[9/16] w-full max-w-[300px] overflow-hidden rounded-(--radius-frame) border border-line bg-panel">
              {clip.urls ? (
                <video ref={videoRef} key={clip.version} controls playsInline preload="metadata" poster={mediaUrl(clip.urls.thumbnail)} src={mediaUrl(clip.urls.video)} className="size-full object-cover" aria-label="Rendered clip preview" />
              ) : (
                <div className="flex size-full items-center justify-center p-6 text-center text-sm text-faint">No rendered file yet</div>
              )}
              {rendering && (
                <div className="absolute inset-x-0 bottom-0 bg-black/90 p-4 text-sm" aria-live="polite">
                  {clip.status === "queued" ? "Waiting for a free worker" : `Rendering, ${Math.round(clip.progress * 100)}%`}
                  <div className="mt-2">
                    <ProgressBar value={clip.progress} label="Render progress" indeterminate={clip.status === "queued"} />
                  </div>
                </div>
              )}
            </div>
            <div className="mt-4 flex items-end justify-between">
              <div>
                <p className="text-sm text-muted">
                  Clip {clip.rank}, version {clip.version}
                </p>
                <ScoreNumber clip={clip} size="md" />
              </div>
              {clip.urls && (
                <a href={mediaUrl(clip.urls.download)} className={buttonClass("secondary", "sm")}>
                  <Download className="size-3.5" aria-hidden />
                  MP4
                </a>
              )}
            </div>
            <p className="mt-3 text-[13px] text-faint">
              The preview is the rendered file itself. Changes appear after you render a new version.
              {clip.layouts.length > 0 && ` Current framing: ${clip.layouts.map(layoutName).join(" + ").toLowerCase()}.`}
            </p>
          </aside>

          <form
            className="space-y-12"
            noValidate /* clip boundaries are validated here and by the API; native step checks reject valid times like 18.369 */
            onSubmit={(e) => {
              e.preventDefault();
              void save();
            }}
          >
            <h1 className="text-3xl font-semibold">Edit this clip</h1>
            {clip.error && <Notice tone={clip.status === "failed" ? "error" : "warn"}>{clip.error}</Notice>}

            <section aria-labelledby="trim-h" className="space-y-6">
              <h2 id="trim-h" className="text-2xl font-semibold">
                Start and end
              </h2>
              <div className="flex flex-wrap gap-8">
                <TimeField label="Start" value={draft.start} min={0} max={Math.max(0, draft.end - 3)} onChange={(v) => setD("start", v)} />
                <TimeField label="End" value={draft.end} min={draft.start + 3} max={sourceDuration} onChange={(v) => setD("end", v)} />
              </div>
              <p className="tnum text-sm text-muted">
                {(draft.end - draft.start).toFixed(1)} s selected from the source
                {trimmed && (
                  <>
                    {" "}
                    <button type="button" className="underline underline-offset-4 hover:text-fg" onClick={() => setDraft((d) => (d ? { ...d, start: clip.start, end: clip.end } : d))}>
                      Reset
                    </button>
                  </>
                )}
              </p>
              {lengthProblem && <Notice tone="error">{lengthProblem}</Notice>}
              <div className="max-w-xl border-y border-line">
                <Toggle label="Shorten long pauses" hint="Only silence is removed; reactions and music between words are kept." checked={draft.smartTrim} onChange={(v) => setD("smartTrim", v)} />
              </div>
            </section>

            <section aria-labelledby="framing-h" className="space-y-4 border-t border-line pt-10">
              <h2 id="framing-h" className="text-2xl font-semibold">
                Framing
              </h2>
              <Segmented
                label="How the picture is fitted to 9:16"
                value={draft.framing}
                onChange={(v) => setD("framing", v)}
                options={[
                  { value: "auto", label: "Automatic" },
                  { value: "face", label: "Follow the speaker" },
                  { value: "split", label: "Split screen" },
                  { value: "blur", label: "Blurred backdrop" },
                  { value: "center", label: "Centre crop" },
                ]}
              />
              {clip.framing_notes.map((n) => (
                <p key={n} className="text-[13px] text-faint">
                  {n}
                </p>
              ))}
            </section>

            <section aria-labelledby="captions-h" className="space-y-6 border-t border-line pt-10">
              <h2 id="captions-h" className="text-2xl font-semibold">
                Captions
              </h2>
              {!clip.has_srt ? (
                <p className="text-muted">No speech was recognised in this clip, so there is nothing to caption.</p>
              ) : (
                <>
                  <div className="max-w-xl border-y border-line">
                    <Toggle label="Burn captions into the video" hint="Off keeps the picture clean. The SRT file is available either way." checked={draft.captions.enabled} onChange={(v) => setD("captions", { ...draft.captions, enabled: v })} />
                  </div>
                  {draft.captions.enabled && <CaptionControls value={draft.captions} onChange={(c) => setD("captions", c)} />}

                  <div>
                    <h3 className="text-lg font-semibold">Fix the words</h3>
                    <p className="mt-1 text-sm text-muted">Correct anything that was misheard. Clear a line to remove it. Press a time to hear that line.</p>
                    {trimmed && <p className="mt-2 text-[13px] text-amber">You changed the start or end, so these lines will be regrouped after the next render.</p>}
                    <ol className="mt-4 max-h-[26rem] space-y-1.5 overflow-y-auto pr-2">
                      {(clip.pages ?? []).map((p, i) => (
                        <li key={`${clip.version}-${i}`} className="grid grid-cols-[4.5rem_1fr] items-center gap-3">
                          <button type="button" onClick={() => seekTo(p.start)} className="tnum rounded-md py-1.5 text-left text-[13px] text-faint hover:text-amber" aria-label={`Play from ${timecode(p.start, true)}`}>
                            {timecode(p.start, true)}
                          </button>
                          <input
                            type="text"
                            value={draft.pages[i] ?? p.text}
                            maxLength={400}
                            onChange={(e) => setD("pages", { ...draft.pages, [i]: e.target.value })}
                            aria-label={`Caption at ${timecode(p.start, true)}`}
                            className="h-9 w-full rounded-md border border-line bg-panel px-3 text-sm text-fg focus:border-line-strong"
                          />
                        </li>
                      ))}
                    </ol>
                  </div>
                </>
              )}
            </section>

            {saveError && <Notice tone="error">{saveError}</Notice>}

            <div className="fixed inset-x-0 bottom-0 z-30 border-t border-line bg-black/90 backdrop-blur">
              <div className="mx-auto flex max-w-[1240px] items-center justify-between gap-4 px-5 py-4 sm:px-8">
                <p className="hidden text-sm text-muted sm:block" aria-live="polite">
                  {rendering ? "Rendering your new version…" : changes ? "You have changes that are not rendered yet." : "No changes since the last render."}
                </p>
                <div className="flex w-full gap-3 sm:w-auto">
                  {changes && !rendering && (
                    <Button variant="ghost" onClick={() => setDraft(draftFrom(clip))}>
                      Discard
                    </Button>
                  )}
                  <Button type="submit" variant="primary" size="lg" disabled={!changes || rendering || saving || Boolean(lengthProblem)} className="flex-1 sm:flex-none">
                    {rendering ? "Rendering…" : "Render new version"}
                  </Button>
                </div>
              </div>
            </div>
          </form>
        </div>
      </main>
    </>
  );
}

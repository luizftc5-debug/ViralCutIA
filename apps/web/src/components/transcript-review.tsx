"use client";

import { useEffect, useMemo, useState } from "react";

import { api, type Transcript, type TranscriptWord } from "@/lib/api";
import { timecode } from "@/lib/format";

import { Button, Notice } from "./ui";

interface Line {
  start: number;
  words: TranscriptWord[];
}

/** Lines of at most 12 words, broken at pauses, so a person can read and fix the transcript. */
function toLines(words: TranscriptWord[]): Line[] {
  const lines: Line[] = [];
  let cur: TranscriptWord[] = [];
  const flush = () => {
    const first = cur[0];
    if (first) lines.push({ start: first.start, words: cur });
    cur = [];
  };
  for (const w of words) {
    const prev = cur[cur.length - 1];
    if (prev && (cur.length >= 12 || w.start - prev.end > 0.9 || (/[.?!…]$/.test(prev.text) && cur.length >= 6))) flush();
    cur.push(w);
  }
  flush();
  return lines;
}

/**
 * Review step: read what the recogniser heard and correct it before anything is rendered.
 * Edits change text only; every word keeps the time it was spoken at.
 */
export function TranscriptReview({ jobId, onContinue, busy }: { jobId: string; onContinue: () => void; busy: boolean }) {
  const [data, setData] = useState<Transcript | null>(null);
  const [drafts, setDrafts] = useState<Record<number, string>>({});
  const [find, setFind] = useState("");
  const [replace, setReplace] = useState("");
  const [error, setError] = useState("");
  const [note, setNote] = useState("");
  const [saving, setSaving] = useState(false);

  const load = () =>
    api
      .transcript(jobId)
      .then((t) => {
        setData(t);
        setDrafts({});
      })
      .catch((e) => setError(e instanceof Error ? e.message : "Could not load the transcript."));
  useEffect(() => {
    void load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  const lines = useMemo(() => (data ? toLines(data.words) : []), [data]);
  const dirty = Object.keys(drafts).length > 0;

  const save = async () => {
    setError("");
    setNote("");
    const edits: { index: number; text: string }[] = [];
    for (const [i, text] of Object.entries(drafts)) {
      const line = lines[Number(i)];
      if (!line) continue;
      const tokens = text.trim().split(/\s+/).filter(Boolean);
      if (tokens.length !== line.words.length) {
        setError(`Line at ${timecode(line.start)}: keep the same number of words (${line.words.length}). To add or remove words, use Find and replace below.`);
        return;
      }
      line.words.forEach((w, k) => {
        const t = tokens[k];
        if (t !== undefined && w.text !== t) edits.push({ index: w.index, text: t });
      });
    }
    await submit({ words: edits });
  };

  const submit = async (body: Parameters<typeof api.saveTranscript>[1]) => {
    setSaving(true);
    try {
      const r = await api.saveTranscript(jobId, body);
      setNote(r.changed ? `Saved ${r.changed} correction${r.changed === 1 ? "" : "s"}.` : "Nothing needed changing.");
      await load();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not save the corrections.");
    } finally {
      setSaving(false);
    }
  };

  if (!data) return <p className="mt-6 text-muted">{error || "Loading transcript…"}</p>;
  const dropped = data.quality.dropped_segments ?? [];

  return (
    <div className="mt-6 max-w-3xl">
      <div className="flex flex-wrap gap-2">
        <input
          aria-label="Find"
          value={find}
          onChange={(e) => setFind(e.target.value)}
          placeholder="Find (whole word or phrase)"
          className="h-10 min-w-0 flex-1 rounded-lg border border-line bg-panel px-3 text-fg placeholder:text-faint"
        />
        <input
          aria-label="Replace with"
          value={replace}
          onChange={(e) => setReplace(e.target.value)}
          placeholder="Replace with"
          className="h-10 min-w-0 flex-1 rounded-lg border border-line bg-panel px-3 text-fg placeholder:text-faint"
        />
        <Button
          disabled={!find.trim() || saving}
          onClick={async () => {
            await submit({ replace: [{ find: find.trim(), replace: replace.trim() }] });
            setFind("");
            setReplace("");
          }}
        >
          Replace everywhere
        </Button>
      </div>

      <ul className="mt-5 divide-y divide-line rounded-xl border border-line bg-panel">
        {lines.map((line, i) => {
          const firstWord = line.words[0];
          if (!firstWord) return null;
          const text = line.words.map((w) => w.text).join(" ");
          const low = line.words.some((w) => w.confidence < 0.5);
          return (
            <li key={firstWord.index} className="flex items-start gap-3 px-3 py-2">
              <span className="tnum mt-2 w-12 shrink-0 text-[13px] text-faint">{timecode(line.start)}</span>
              <input
                aria-label={`Transcript line at ${timecode(line.start)}`}
                value={drafts[i] ?? text}
                spellCheck
                lang={data.language || undefined}
                onChange={(e) => setDrafts((d) => (e.target.value === text ? Object.fromEntries(Object.entries(d).filter(([k]) => Number(k) !== i)) : { ...d, [i]: e.target.value }))}
                className="h-9 min-w-0 flex-1 rounded-md border border-transparent bg-transparent px-2 text-fg hover:border-line focus:border-amber"
              />
              {low && <span className="mt-2 text-[12px] text-amber" title="The recogniser was unsure about a word on this line">check</span>}
            </li>
          );
        })}
      </ul>

      {dropped.length > 0 && (
        <p className="mt-3 text-[13px] text-faint">
          {dropped.length} stretch{dropped.length === 1 ? "" : "es"} of non-speech (music, noise or silence) were left out instead of being guessed at.
        </p>
      )}
      {error && (
        <div className="mt-4">
          <Notice tone="error">{error}</Notice>
        </div>
      )}
      {note && <p className="mt-3 text-sm text-muted">{note}</p>}
      <div className="mt-6 flex flex-wrap items-center gap-3">
        <Button disabled={!dirty || saving} onClick={() => void save()}>
          Save line edits
        </Button>
        <Button variant="primary" disabled={dirty || saving || busy} onClick={onContinue}>
          Continue and render clips
        </Button>
        {dirty && <span className="text-[13px] text-faint">Save your edits before continuing.</span>}
      </div>
    </div>
  );
}

"use client";

import type { Growth } from "@/lib/api";
import { timecode } from "@/lib/format";

import { CopyButton } from "./ui";

/** Creator Growth Toolkit: publishing material quoted from the clip itself. No view promises. */
export function GrowthToolkit({ g }: { g: Growth }) {
  const chapters = g.chapters && g.chapters.length >= 3 ? g.chapters.map((c) => `${timecode(c.t)} ${c.title}`).join("\n") : "";
  return (
    <details className="mt-6 max-w-2xl">
      <summary className="text-sm text-muted hover:text-fg">Growth toolkit: titles, description, keywords{chapters ? ", chapters" : ""}</summary>
      <div className="mt-4 space-y-5 text-[14px]">
        {g.hook && (
          <div>
            <h4 className="text-[13px] text-faint">Opening hook (spoken in the clip)</h4>
            <p className="mt-1">“{g.hook}”</p>
          </div>
        )}
        <div>
          <h4 className="text-[13px] text-faint">Description</h4>
          <div className="mt-1 flex items-start justify-between gap-3">
            <p className="whitespace-pre-line">{g.description}</p>
            <CopyButton text={g.description} label="Copy description" />
          </div>
        </div>
        {g.keywords.length > 0 && (
          <div>
            <h4 className="text-[13px] text-faint">Keywords</h4>
            <p className="mt-1">{g.keywords.join(", ")}</p>
          </div>
        )}
        {g.hashtags.length > 0 && (
          <div className="flex items-center justify-between gap-3">
            <p>{g.hashtags.join(" ")}</p>
            <CopyButton text={g.hashtags.join(" ")} label="Copy hashtags" />
          </div>
        )}
        {g.thumbnail_text.length > 0 && (
          <div>
            <h4 className="text-[13px] text-faint">Thumbnail text ideas</h4>
            <ul className="mt-1 flex flex-wrap gap-2">
              {g.thumbnail_text.map((t) => (
                <li key={t} className="rounded border border-line px-2 py-1 font-display text-sm font-semibold">
                  {t}
                </li>
              ))}
            </ul>
          </div>
        )}
        {Object.keys(g.platforms).length > 0 && (
          <div>
            <h4 className="text-[13px] text-faint">Ready to paste, per platform</h4>
            <ul className="mt-1 divide-y divide-line border-y border-line">
              {Object.entries(g.platforms).map(([k, v]) => (
                <li key={k} className="flex items-start justify-between gap-3 py-2">
                  <span>
                    <span className="mr-2 text-faint">{k.replace("_", " ")}</span>
                    {v}
                  </span>
                  <CopyButton text={v} label={`Copy for ${k}`} />
                </li>
              ))}
            </ul>
          </div>
        )}
        {chapters && (
          <div>
            <div className="flex items-center justify-between">
              <h4 className="text-[13px] text-faint">YouTube chapters (from topic changes)</h4>
              <CopyButton text={chapters} label="Copy chapters" />
            </div>
            <pre className="mt-1 whitespace-pre-wrap font-sans">{chapters}</pre>
          </div>
        )}
        {g.followups.length > 0 && (
          <div>
            <h4 className="text-[13px] text-faint">Follow-up ideas (other strong moments of this video)</h4>
            <ul className="mt-1 list-disc pl-5">
              {g.followups.map((f) => (
                <li key={f}>{f}</li>
              ))}
            </ul>
          </div>
        )}
        {g.versions_to_test.length > 0 && <p className="text-faint">Versions to compare: {g.versions_to_test.join(", ")}</p>}
        <p className="text-[12px] text-faint">{g.note}</p>
      </div>
    </details>
  );
}

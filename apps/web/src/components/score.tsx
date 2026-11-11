import type { Clip } from "@/lib/api";

import { cx } from "./ui";

const ENGINE_NOTE: Record<Clip["engine"], string> = {
  heuristic: "Estimated from speech, audio and picture signals.",
  "llm+signals": "Estimated by a language model reading the transcript, blended with audio and picture signals.",
  "signals-only": "Low confidence: no speech was found, so only audio and picture signals were used.",
};

export function ScoreNumber({ clip, size = "lg" }: { clip: Clip; size?: "lg" | "md" }) {
  return (
    <p className="flex items-baseline gap-1" title={ENGINE_NOTE[clip.engine]}>
      <span className={cx("tnum font-display font-semibold leading-none text-amber", size === "lg" ? "text-5xl" : "text-3xl")}>{Math.round(clip.score)}</span>
      <span className="tnum text-sm text-faint">/ 100</span>
      <span className="sr-only">Viral potential score. {ENGINE_NOTE[clip.engine]}</span>
    </p>
  );
}

/** The six dimensions behind a score, each with its weight. */
export function ScoreBreakdown({ clip }: { clip: Clip }) {
  return (
    <div>
      <dl className="space-y-2.5">
        {clip.dimensions.map((d) => (
          <div key={d.key} className="grid grid-cols-[1fr_auto] items-center gap-x-3 text-[13px]">
            <dt className="text-muted">
              {d.label} <span className="tnum text-faint">({Math.round(d.weight * 100)}% of the score)</span>
            </dt>
            <dd className="tnum text-fg">{d.value}</dd>
            <div className="col-span-2 h-1 overflow-hidden rounded-full bg-line" aria-hidden>
              <div className="h-full rounded-full bg-amber/80" style={{ width: `${d.value}%` }} />
            </div>
          </div>
        ))}
      </dl>
      <p className="mt-3 text-[13px] text-faint">
        {ENGINE_NOTE[clip.engine]} The score is an estimate of how well this moment opens, holds and stands alone. It is not a prediction of views.
      </p>
    </div>
  );
}

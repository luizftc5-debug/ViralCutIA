import Image from "next/image";
import Link from "next/link";

import { CAPTION_STYLES } from "@/lib/caption-styles";
import { RecentProjects } from "@/components/recent-projects";
import { SiteHeader, Wordmark } from "@/components/site-header";
import { TimelineDropzone } from "@/components/timeline-dropzone";

// The scoring weights shipped in the engine (apps/api/viralcut/pipeline/scoring.py).
const DIMENSIONS = [
  { label: "Hook strength", weight: 25, detail: "Do the first three seconds give a reason to keep watching? A question, a surprising claim, real stakes, an immediate promise." },
  { label: "Narrative retention", weight: 20, detail: "The moment has to develop and pay off, and stay on one idea. Cuts that wander across topics lose points." },
  { label: "Content value", weight: 20, detail: "Specific, concrete sentences count. Filler dilutes the score, and a longer clip does not earn more by being longer." },
  { label: "Emotional and visual impact", weight: 15, detail: "Charged language, vocal dynamics measured from the audio, and motion measured from the picture." },
  { label: "Standalone clarity", weight: 10, detail: "A clip that opens on “and then he said” needs the rest of the video. Those are pushed down." },
  { label: "Technical quality", weight: 10, detail: "Audio level, noise, transcription confidence and source resolution." },
];

const DELIVERABLES = [
  ["Vertical MP4, ready to post", "1080 × 1920, H.264 and AAC, original frame rate, fast-start. Every file is probed and fully decoded before it is shown as ready."],
  ["Framing that follows the speaker", "Faces are tracked shot by shot on a smoothed path. Two people who can’t share a vertical frame get a split screen; footage with no one to follow keeps the whole frame over a blurred backdrop."],
  ["Captions timed to each word", `${CAPTION_STYLES.length} styles, burned in or left off. Lines break where the speaker pauses and stay clear of each platform’s buttons. An SRT file comes with every clip.`],
  ["Tighter pacing", "Dead air is shortened only where the audio confirms it is silent, so laughter and reactions survive, and a beat after a question is kept."],
  ["Titles and a caption to post with", "Taken from what was actually said in the clip. Nothing is invented to make it sound bigger than it is."],
  ["A small studio for the last 5%", "Move the in and out points, fix a misheard word, change the caption style or framing, and render a new version."],
];

export default function LandingPage() {
  return (
    <>
      <SiteHeader />
      <main id="main">
        <section className="mx-auto max-w-[1240px] px-5 pb-20 pt-16 sm:px-8 sm:pt-24">
          <h1 className="max-w-[15ch] text-[clamp(2.75rem,7.4vw,6rem)] font-bold leading-[0.98]">
            Turn long videos into content worth watching.
          </h1>
          <p className="mt-6 max-w-xl text-lg text-muted sm:text-xl">Upload once. Discover your best moments. Publish everywhere.</p>
          <div className="mt-10 sm:mt-14">
            <TimelineDropzone />
          </div>
          <p className="mt-4 max-w-2xl text-sm text-faint">
            Use footage you own or are allowed to edit. Uploads are private to your link and are deleted automatically after the retention period.
          </p>
          <RecentProjects />
        </section>

        <section id="how-it-picks" className="border-t border-line">
          <div className="mx-auto grid max-w-[1240px] gap-x-16 gap-y-10 px-5 py-20 sm:px-8 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div>
              <h2 className="text-3xl font-semibold sm:text-4xl">It reads the video before it cuts it.</h2>
              <p className="mt-5 text-muted">
                ViralCut does not slice your video into equal blocks. It transcribes the speech with word-level timing, listens to the
                audio, watches for shot changes, and then proposes clips that begin at the start of a sentence and end when the thought
                does.
              </p>
              <p className="mt-4 text-muted">
                Each candidate gets a score from 0 to 100 built from six measurements. Near-duplicates are removed, and if a video only
                has three good moments you get three clips, not ten.
              </p>
            </div>

            <div>
              <div className="flex h-3 overflow-hidden rounded-full" role="img" aria-label="Score weights: hook 25%, narrative 20%, content 20%, impact 15%, clarity 10%, technical 10%">
                {DIMENSIONS.map((d, i) => (
                  <span key={d.label} className="h-full border-r-2 border-black bg-amber last:border-r-0" style={{ width: `${d.weight}%`, opacity: 1 - i * 0.13 }} />
                ))}
              </div>
              <dl className="mt-6 divide-y divide-line">
                {DIMENSIONS.map((d) => (
                  <div key={d.label} className="grid grid-cols-[3.5rem_1fr] gap-x-4 py-4">
                    <dt className="col-start-2 font-medium">{d.label}</dt>
                    <dd className="tnum col-start-1 row-start-1 font-display text-2xl font-semibold text-amber">{d.weight}%</dd>
                    <dd className="col-start-2 mt-1 text-sm text-muted">{d.detail}</dd>
                  </div>
                ))}
              </dl>
            </div>
          </div>
        </section>

        <section className="border-t border-line">
          <div className="mx-auto max-w-[1240px] px-5 py-20 sm:px-8">
            <h2 className="max-w-2xl text-3xl font-semibold sm:text-4xl">What comes back for each clip</h2>
            <dl className="mt-10 grid gap-x-12 gap-y-9 sm:grid-cols-2 lg:grid-cols-3">
              {DELIVERABLES.map(([title, body]) => (
                <div key={title} className="border-t border-line-strong pt-4">
                  <dt className="font-medium">{title}</dt>
                  <dd className="mt-2 text-sm text-muted">{body}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <section id="captions" className="border-t border-line">
          <div className="mx-auto max-w-[1240px] px-5 py-20 sm:px-8">
            <div className="max-w-2xl">
              <h2 className="text-3xl font-semibold sm:text-4xl">Captions that help people follow, in {CAPTION_STYLES.length} styles</h2>
              <p className="mt-5 text-muted">
                These frames were rendered by the same engine that burns captions into your clips. Change the accent colour, size,
                position and typeface, or turn captions off entirely.
              </p>
            </div>
            <ul className="mt-10 grid gap-5 sm:grid-cols-2">
              {CAPTION_STYLES.map((s) => (
                <li key={s.key}>
                  <Image src={`/caption-styles/${s.key}.webp`} alt={`Example of the ${s.label} caption style`} width={720} height={374} className="w-full rounded-(--radius-frame) border border-line" />
                  <p className="mt-3 font-medium">{s.label}</p>
                  <p className="text-sm text-muted">{s.description}</p>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section className="border-t border-line">
          <div className="mx-auto max-w-[1240px] px-5 py-20 sm:px-8">
            <div className="max-w-2xl">
              <h2 className="text-3xl font-semibold sm:text-4xl">What the score is, and what it isn’t</h2>
              <p className="mt-5 text-muted">
                Nobody can promise that a clip will go viral, and ViralCut does not try. The score is an estimate of how well a moment
                opens, holds and stands on its own. Its weights are starting hypotheses, not a formula taken from any platform, and
                no view counts or probabilities are predicted.
              </p>
              <p className="mt-4 text-muted">
                When the analysis has less to go on, such as footage without speech, or a server running without the optional
                language model, the results say so next to the score.
              </p>
              <Link href="/new" className="mt-8 inline-flex h-12 items-center rounded-lg bg-fg px-6 font-medium text-black hover:bg-white">
                Upload a video
              </Link>
            </div>
          </div>
        </section>
      </main>

      <footer className="border-t border-line">
        <div className="mx-auto flex max-w-[1240px] flex-col gap-3 px-5 py-8 text-sm text-faint sm:flex-row sm:items-center sm:justify-between sm:px-8">
          <Wordmark />
          <p>Open source under the MIT licence. Process only video you have the rights to.</p>
        </div>
      </footer>
    </>
  );
}

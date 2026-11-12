import type { CaptionStyle } from "./api";

/** Mirrors the styles defined by the caption engine (apps/api/viralcut/pipeline/captions.py). */
export const CAPTION_STYLES: { key: Exclude<CaptionStyle, "auto">; label: string; description: string }[] = [
  { key: "viral_dynamic", label: "Bold Dynamic", description: "High-contrast captions: the spoken word lights up and keywords are emphasised in sequence." },
  { key: "energy", label: "High Energy", description: "Large expressive words that pop with the voice; thick outline, controlled tilt." },
  { key: "podcast", label: "Podcast Premium", description: "Clean sentence-case lines on a soft panel, subtle highlight on the spoken word." },
  { key: "kinetic", label: "Kinetic Typography", description: "Short phrases slide in and every word animates as it is spoken." },
  { key: "clean_editorial", label: "Minimal Editorial", description: "Quiet, elegant lines that reveal as they are spoken. Restrained movement." },
  { key: "gaming", label: "Gaming Impact", description: "High-contrast words that punch in with the commentary, placed clear of the gameplay." },
  { key: "cinematic", label: "Cinematic", description: "Refined serif type with a soft shadow; words fade in gently, no bounce." },
  { key: "high_retention", label: "Classic Viral", description: "High-visibility word-by-word phrases; a colour highlight follows the voice." },
  { key: "dynamic_creator", label: "Highlight Box", description: "A highlight block follows the voice from word to word." },
];

export const VIDEO_STYLE_LABELS: Record<string, string> = {
  auto: "Automatic AI",
  viral: "Viral Fast-Paced",
  podcast: "Podcast Clips",
  gaming: "Gaming Highlights",
  cinematic: "Cinematic Storytelling",
  educational: "Educational / Authority",
  reaction: "Reaction / Entertainment",
  minimal: "Minimal Premium",
  storytime: "Storytime",
  news: "News / Commentary",
  dynamic: "Viral Fast-Paced",
  editorial: "Minimal Premium",
};

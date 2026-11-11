import type { CaptionStyle } from "./api";

/** Mirrors the styles defined by the caption engine (apps/api/viralcut/pipeline/captions.py). */
export const CAPTION_STYLES: { key: CaptionStyle; label: string; description: string }[] = [
  { key: "high_retention", label: "High Retention", description: "Bold word-by-word captions. The spoken word lights up." },
  { key: "clean_editorial", label: "Clean Editorial", description: "Quiet, elegant lines that reveal as they are spoken." },
  { key: "dynamic_creator", label: "Dynamic Creator", description: "A highlight block follows the voice from word to word." },
  { key: "gaming", label: "Gaming / Entertainment", description: "Oversized, high-contrast words that punch in with the commentary." },
];

import type { CaptionStyle } from "./api";

/** Mirrors the styles defined by the caption engine (apps/api/viralcut/pipeline/captions.py). */
export const CAPTION_STYLES: { key: CaptionStyle; label: string; description: string }[] = [
  { key: "viral_dynamic", label: "Viral Dynamic", description: "Bold, clean captions: the spoken word lights up, keywords stand out, phrases slide in." },
  { key: "high_retention", label: "Creator High Retention", description: "Short word-by-word phrases; a controlled colour highlight follows the voice." },
  { key: "clean_editorial", label: "Premium Minimal", description: "Quiet, elegant lines that reveal as they are spoken. Restrained movement." },
  { key: "dynamic_creator", label: "Dynamic Creator", description: "A highlight block follows the voice from word to word." },
  { key: "gaming", label: "Gaming / Entertainment", description: "High-contrast words that punch in with the commentary, placed clear of the gameplay." },
];

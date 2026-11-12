import { describe, expect, it } from "vitest";

import { DEFAULT_OPTIONS, type PresetInfo } from "./api";
import { applyPreset, summarize, switchPreset } from "./presets";

const AUTO: PresetInfo = { key: "auto", label: "AI Automatic", description: "", options: { style: "auto", intensity: "balanced", captions: { style: "auto" } }, editors: {} };
const FACELESS: PresetInfo = {
  key: "faceless",
  label: "Faceless YouTube",
  description: "",
  options: { style: "educational", captions: { style: "kinetic" }, framing: "blur" },
  editors: { animacao: { enabled: true }, "3d": { enabled: true, strength: "subtle" } },
};
const PODCAST: PresetInfo = {
  key: "podcast",
  label: "Podcast Viral Clips",
  description: "",
  options: { style: "podcast", captions: { style: "podcast" } },
  editors: { corte: { remove_fillers: true } },
};

describe("presets", () => {
  it("applies a preset on top of the options", () => {
    const o = applyPreset(DEFAULT_OPTIONS, FACELESS);
    expect(o.preset).toBe("faceless");
    expect(o.framing).toBe("blur");
    expect(o.captions.style).toBe("kinetic");
    expect(o.captions.enabled).toBe(DEFAULT_OPTIONS.captions.enabled);
    expect(Object.keys(o.editors).sort()).toEqual(["3d", "animacao"]);
  });

  it("leaves nothing of the previous preset behind when another is picked", () => {
    const first = applyPreset(DEFAULT_OPTIONS, FACELESS);
    const second = switchPreset(first, PODCAST, FACELESS);
    expect(second.preset).toBe("podcast");
    expect(second.framing).toBe(DEFAULT_OPTIONS.framing);
    expect(second.captions.style).toBe("podcast");
    expect(Object.keys(second.editors)).toEqual(["corte"]);
    const back = switchPreset(second, AUTO, PODCAST);
    expect(back.editors).toEqual({});
    expect(back.style).toBe("auto");
  });

  it("keeps what the user set by hand", () => {
    const mine = { ...applyPreset(DEFAULT_OPTIONS, FACELESS), clip_count: 3 as const, language: "pt", editors: { ...FACELESS.editors, trilha: { enabled: true } } };
    const next = switchPreset(mine, PODCAST, FACELESS);
    expect(next.clip_count).toBe(3);
    expect(next.language).toBe("pt");
    expect(next.editors.trilha).toEqual({ enabled: true });
  });

  it("says what will be generated", () => {
    expect(summarize(DEFAULT_OPTIONS, [AUTO])).toBe("YouTube Shorts (9:16) · AI Automatic · captions on");
    const o = { ...applyPreset(DEFAULT_OPTIONS, PODCAST), content_type: "youtube" as const, clip_count: 1 as const, captions: { ...DEFAULT_OPTIONS.captions, enabled: false } };
    expect(summarize(o, [AUTO, PODCAST])).toBe("YouTube long-form (16:9) · Podcast Viral Clips · no captions · up to 1 clip");
  });
});

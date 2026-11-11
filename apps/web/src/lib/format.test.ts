import { describe, expect, it } from "vitest";

import { fileSize, languageName, layoutName, seconds, seeded, timecode } from "./format";

describe("format", () => {
  it("formats timecodes", () => {
    expect(timecode(0)).toBe("0:00");
    expect(timecode(83.4)).toBe("1:23");
    expect(timecode(3723)).toBe("1:02:03");
    expect(timecode(83.46, true)).toBe("1:23.5");
    expect(timecode(undefined)).toBe("–");
  });

  it("formats sizes and durations", () => {
    expect(fileSize(0)).toBe("–");
    expect(fileSize(900)).toBe("900 B");
    expect(fileSize(5.6 * 1024 * 1024)).toBe("5.6 MB");
    expect(seconds(23.2)).toBe("23s");
    expect(seconds(75)).toBe("1:15");
  });

  it("names languages and layouts", () => {
    expect(languageName("pt")).toBe("Portuguese");
    expect(languageName("xx")).toBe("XX");
    expect(layoutName("blur")).toBe("Blurred background");
  });

  it("seeded noise is deterministic and bounded", () => {
    expect(seeded(7)).toBe(seeded(7));
    for (let i = 0; i < 50; i++) {
      expect(seeded(i)).toBeGreaterThanOrEqual(0);
      expect(seeded(i)).toBeLessThan(1);
    }
  });
});

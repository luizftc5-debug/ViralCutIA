import { describe, expect, it } from "vitest";

import type { Stage } from "./api";
import { isActive, projectStatus } from "./status";

const stages: Stage[] = [
  { key: "analyzing", label: "Analyzing video", status: "done", progress: 1 },
  { key: "transcribing", label: "Transcribing speech", status: "active", progress: 0.2 },
];

describe("project status", () => {
  it("says what a running video is doing", () => {
    expect(projectStatus({ status: "processing", progress: 0.156, stages, clip_count: 0 })).toEqual({ tone: "busy", text: "Processing 16% · Transcribing speech" });
    expect(projectStatus({ status: "queued", progress: 0, stages: [], clip_count: 0 }).text).toBe("In the queue");
    expect(isActive({ status: "processing" }) && isActive({ status: "queued" }) && !isActive({ status: "failed" })).toBe(true);
  });

  it("tells a finished video with clips from one without", () => {
    expect(projectStatus({ status: "completed", progress: 1, stages: [], clip_count: 1 })).toEqual({ tone: "ok", text: "Ready · 1 clip" });
    expect(projectStatus({ status: "completed", progress: 1, stages: [], clip_count: 5 }).text).toBe("Ready · 5 clips");
    expect(projectStatus({ status: "completed", progress: 1, stages: [], clip_count: 0 }).tone).toBe("idle");
    expect(projectStatus({ status: "failed", progress: 0.3, stages: [], clip_count: 0 }).tone).toBe("bad");
  });
});

import type { JobSummary } from "./api";

export type StatusTone = "ok" | "busy" | "bad" | "idle";

/** What a project is doing, in a few words, for lists (the job page says it in full). */
export function projectStatus(job: Pick<JobSummary, "status" | "progress" | "stages" | "clip_count">): { tone: StatusTone; text: string } {
  switch (job.status) {
    case "processing": {
      const stage = job.stages?.find((s) => s.status === "active");
      return { tone: "busy", text: `Processing ${Math.round((job.progress ?? 0) * 100)}%${stage ? ` · ${stage.label}` : ""}` };
    }
    case "queued":
      return { tone: "busy", text: "In the queue" };
    case "review":
      return { tone: "busy", text: "Waiting for you: check the transcript" };
    case "completed":
      return job.clip_count > 0 ? { tone: "ok", text: `Ready · ${job.clip_count} clip${job.clip_count === 1 ? "" : "s"}` } : { tone: "idle", text: "Finished · no moment made the cut" };
    case "failed":
      return { tone: "bad", text: "Stopped with an error" };
    case "canceled":
      return { tone: "idle", text: "Canceled" };
    default:
      return { tone: "idle", text: "Uploaded, not processed yet" };
  }
}

export const isActive = (job: Pick<JobSummary, "status">) => job.status === "queued" || job.status === "processing";

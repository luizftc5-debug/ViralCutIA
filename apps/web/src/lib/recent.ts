/** Recent projects live only in this browser: the server has no accounts. */

export interface RecentJob {
  id: string;
  filename: string;
  createdAt: string;
}

const KEY = "viralcut.recent";

export function listRecent(): RecentJob[] {
  if (typeof window === "undefined") return [];
  try {
    const parsed = JSON.parse(window.localStorage.getItem(KEY) ?? "[]") as RecentJob[];
    return Array.isArray(parsed) ? parsed.filter((j) => /^[0-9a-f]{32}$/.test(j.id)) : [];
  } catch {
    return [];
  }
}

function save(items: RecentJob[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(items.slice(0, 12)));
    window.dispatchEvent(new Event("viralcut:recent"));
  } catch {
    /* storage unavailable (private mode): recents are a convenience only */
  }
}

export function rememberJob(job: RecentJob): void {
  save([job, ...listRecent().filter((j) => j.id !== job.id)]);
}

export function forgetJob(id: string): void {
  save(listRecent().filter((j) => j.id !== id));
}

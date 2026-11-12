import Link from "next/link";

import { buttonClass } from "@/lib/classes";

/** Wordmark: the name between in/out marks, the way an editor brackets a selection. */
export function Wordmark() {
  return (
    <Link href="/" className="group inline-flex items-center gap-1.5 font-display text-[19px] font-semibold tracking-tight" aria-label="ViralCut AI, home">
      <svg viewBox="0 0 10 20" className="h-5 w-2.5 text-amber" aria-hidden>
        <path d="M9 1H2v18h7" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
      <span>ViralCut</span>
      <svg viewBox="0 0 10 20" className="h-5 w-2.5 text-amber" aria-hidden>
        <path d="M1 1h7v18H1" fill="none" stroke="currentColor" strokeWidth="2" />
      </svg>
    </Link>
  );
}

export function SiteHeader({ studio = false }: { studio?: boolean }) {
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-black/85 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-[1240px] items-center justify-between px-5 sm:px-8">
        <Wordmark />
        <nav aria-label="Main" className="flex items-center gap-1 text-sm">
          {!studio && (
            <>
              <a href="#how-it-picks" className="hidden rounded-md px-3 py-1.5 text-muted hover:text-fg sm:block">
                How it picks
              </a>
              <a href="#captions" className="hidden rounded-md px-3 py-1.5 text-muted hover:text-fg sm:block">
                Captions
              </a>
            </>
          )}
          <Link href="/projects" className="rounded-md px-3 py-1.5 text-muted hover:text-fg">
            Projects
          </Link>
          <Link href="/new" className={buttonClass(studio ? "secondary" : "primary", "sm")}>
            {studio ? "New video" : "Open studio"}
          </Link>
        </nav>
      </div>
    </header>
  );
}

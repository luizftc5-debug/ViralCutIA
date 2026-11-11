import Link from "next/link";

import { SiteHeader } from "@/components/site-header";
import { buttonClass } from "@/lib/classes";

export default function NotFound() {
  return (
    <>
      <SiteHeader studio />
      <main id="main" className="mx-auto max-w-[1240px] px-5 py-24 sm:px-8">
        <h1 className="text-4xl font-semibold">That page doesn’t exist</h1>
        <p className="mt-4 max-w-xl text-muted">The address may be mistyped, or the video it pointed to has been deleted.</p>
        <Link href="/" className={`${buttonClass("primary", "lg")} mt-8`}>
          Go to the start
        </Link>
      </main>
    </>
  );
}

import type { Metadata, Viewport } from "next";

import { UploadProvider } from "@/components/upload-provider";

import "./globals.css";

export const metadata: Metadata = {
  title: { default: "ViralCut AI — turn long videos into short clips worth watching", template: "%s | ViralCut AI" },
  description:
    "Upload one long video. ViralCut finds its strongest moments, reframes them to 9:16, adds synced captions and renders clips ready for Shorts, Reels and TikTok.",
  robots: { index: false }, // project pages carry private capability URLs
};

export const viewport: Viewport = { themeColor: "#000000", colorScheme: "dark" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <a href="#main" className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-50 focus:rounded-md focus:bg-fg focus:px-3 focus:py-2 focus:text-black">
          Skip to content
        </a>
        <UploadProvider>{children}</UploadProvider>
      </body>
    </html>
  );
}

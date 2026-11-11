import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // the Docker image sets NEXT_OUTPUT=standalone to get a self-contained server bundle
  output: process.env.NEXT_OUTPUT === "standalone" ? "standalone" : undefined,
  reactStrictMode: true,
  poweredByHeader: false,
  async headers() {
    return [
      {
        source: "/(.*)",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          // job ids in URLs are capabilities: never send them to other origins
          { key: "Referrer-Policy", value: "no-referrer" },
          { key: "X-Frame-Options", value: "DENY" },
        ],
      },
    ];
  },
};

export default nextConfig;

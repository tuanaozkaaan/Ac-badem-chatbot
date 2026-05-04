/**
 * Next.js configuration for the ACUdost frontend.
 *
 * Strategy (Adım 5.2)
 * --------------------
 * The browser only ever talks to the Next.js origin; all chatbot calls
 * go through Route Handlers under `app/api/*`, which forward server-side
 * to ``ACU_BACKEND_URL`` (Django). This keeps:
 *   - CORS off the table (single-origin from the browser's POV),
 *   - the Django session cookie sealed in a server-server hop,
 *   - the Django CSRF model intact (we only call ``@csrf_exempt`` /api/v1/ask).
 *
 * No `images.remotePatterns` are needed because the avatar is a local
 * `public/` asset.
 */
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Surfaces lint failures during ``next build``; strictNullChecks etc. are
  // controlled by tsconfig.json.
  eslint: {
    ignoreDuringBuilds: false,
  },
  experimental: {
    // App Router is the only router we use; staying conservative on flags.
  },
};

export default nextConfig;

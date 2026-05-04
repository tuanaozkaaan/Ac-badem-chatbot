/**
 * Root layout for the ACUdost Next.js port.
 *
 * The visual chrome is intentionally identical to the legacy Django template
 * (templates/index.html). When you change something here, mirror it there
 * and vice versa, otherwise the two surfaces will visually drift.
 *
 * Plus Jakarta Sans is loaded via Next's `next/font/google` integration so
 * the file is self-hosted at build time (no third-party fetch on first paint).
 */
import type { Metadata, Viewport } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";

import "./globals.css";

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin", "latin-ext"],
  weight: ["400", "500", "600", "700"],
  style: ["normal", "italic"],
  display: "swap",
  variable: "--font-plus-jakarta",
});

export const metadata: Metadata = {
  title: "ACUdost — Acıbadem Üniversitesi",
  description:
    "Acıbadem Üniversitesi resmî kaynaklarına dayalı, yerel RAG destekli akademik bilgi asistanı.",
  icons: {
    icon: "/avatar.png",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0c1220",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="tr" className={plusJakartaSans.variable}>
      <body
        // CSS expects a CSS custom property `--app-bg`. The legacy template
        // pointed it at `chatbot/assets/acibadem-bg.png`, but that asset never
        // existed in the repo so the gradient overlay alone is the canonical
        // visual. Setting `none` keeps the appearance identical without
        // emitting a 404 for a missing image.
        style={{ ["--app-bg" as string]: "none" }}
      >
        {children}
      </body>
    </html>
  );
}

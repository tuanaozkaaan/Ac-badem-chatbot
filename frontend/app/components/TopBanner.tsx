"use client";

/**
 * Single top-of-viewport banner that surfaces transient user-facing
 * conditions: browser offline, Django proxy unreachable, or per-IP
 * rate-limit hit (Adım 5.4 — tasks #3 + #4 unified).
 *
 * Rationale for one component
 * ---------------------------
 * Two stacked banners would compete for attention and shift the chat
 * layout twice when both fire. Rolling them up here keeps the UX
 * deterministic: at most one row of warning is visible at a time,
 * with a precedence order (offline > rate_limited > server_unreachable)
 * that matches how the parent should react to the most blocking
 * condition first.
 *
 * Lifecycle
 * ---------
 * * `offline`             — driven by `navigator.onLine`. Sticky until
 *   the browser fires the `online` event.
 * * `rate_limited`        — set by the parent when /api/v1/ask returns
 *   HTTP 429. Auto-dismissed by the parent after a short cool-down (the
 *   banner itself is dumb and just renders what it is told).
 * * `server_unreachable`  — set by the parent when fetch throws or the
 *   Next proxy returns 504/`upstream_unreachable`. Cleared on the next
 *   successful response.
 */
import { useEffect, useState } from "react";

export type TopBannerMode = "offline" | "rate_limited" | "server_unreachable" | null;

type Props = {
  /** Override the auto-detected `navigator.onLine` value (testing aid). */
  forcedMode?: TopBannerMode;
  /** Set by the parent when the API responded 429. */
  rateLimited?: boolean;
  /** Set by the parent on a network/upstream failure that is not "offline". */
  serverUnreachable?: boolean;
};

function copyForMode(mode: TopBannerMode): { headline: string; detail: string } | null {
  switch (mode) {
    case "offline":
      return {
        headline: "Çevrimdışısınız",
        detail:
          "İnternet bağlantınız kesildi. Bağlantı geri geldiğinde mesajlarınızı tekrar gönderebilirsiniz.",
      };
    case "rate_limited":
      return {
        headline: "Çok sık soru soruyorsunuz",
        detail:
          "Hız sınırı aşıldı; lütfen biraz bekleyin. Bu sınır, herkes için arka uçta sıralama gecikmesi olmaması için var.",
      };
    case "server_unreachable":
      return {
        headline: "Bağlantı hatası",
        detail:
          "Sunucuya şu anda ulaşılamıyor. Birkaç saniye içinde aynı soruyu tekrar deneyebilirsiniz.",
      };
    default:
      return null;
  }
}

export default function TopBanner({ forcedMode, rateLimited, serverUnreachable }: Props) {
  const [browserOnline, setBrowserOnline] = useState<boolean>(true);

  useEffect(() => {
    if (forcedMode !== undefined) return;
    setBrowserOnline(navigator.onLine);
    const onOnline = () => setBrowserOnline(true);
    const onOffline = () => setBrowserOnline(false);
    window.addEventListener("online", onOnline);
    window.addEventListener("offline", onOffline);
    return () => {
      window.removeEventListener("online", onOnline);
      window.removeEventListener("offline", onOffline);
    };
  }, [forcedMode]);

  const mode: TopBannerMode = (() => {
    if (forcedMode !== undefined) return forcedMode;
    if (!browserOnline) return "offline";
    if (rateLimited) return "rate_limited";
    if (serverUnreachable) return "server_unreachable";
    return null;
  })();

  const copy = copyForMode(mode);
  if (!copy) return null;

  return (
    <div className={`top-banner top-banner-${mode}`} role="status" aria-live="polite">
      <span className="top-banner-dot" aria-hidden="true" />
      <div className="top-banner-text">
        <strong>{copy.headline}</strong>
        <span>{copy.detail}</span>
      </div>
    </div>
  );
}

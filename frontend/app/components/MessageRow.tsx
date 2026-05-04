"use client";

/**
 * Single row in the message stream.
 *
 * Layouts
 * -------
 * * `assistant`: avatar on the left, bubble + footer in .msg-main, optional
 *   <SourcesCard /> beneath the bubble.
 * * `user`: bubble right-aligned, no avatar (matches templates/index.html).
 */
import type { RetrievedChunk } from "@/app/lib/types";

import SourcesCard from "./SourcesCard";

export type MessageEntry = {
  /** Stable id is required so React reconciles correctly while streams append. */
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Wall-clock time we waited for the assistant turn (assistant only). */
  elapsedMs?: number;
  /** Sources that the v1 backend reported alongside the assistant answer. */
  retrievedChunks?: RetrievedChunk[];
};

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} sn`;
}

export default function MessageRow({ entry }: { entry: MessageEntry }) {
  if (entry.role === "user") {
    return (
      <div className="message-row user">
        <div className="msg-main">
          <div className="bubble user">{entry.text}</div>
        </div>
      </div>
    );
  }

  const showFooter = typeof entry.elapsedMs === "number";
  const showSources = Array.isArray(entry.retrievedChunks) && entry.retrievedChunks.length > 0;

  return (
    <div className="message-row assistant">
      <div className="msg-aside">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="avatar" src="/avatar.png" alt="ACUdost avatar" aria-hidden="true" />
      </div>
      <div className="msg-main">
        <div className="bubble assistant">
          <div
            style={{
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
              lineHeight: 1.55,
              fontSize: "15px",
            }}
          >
            {entry.text}
          </div>
          {showFooter ? (
            <div
              style={{
                marginTop: "12px",
                fontSize: "11px",
                fontWeight: 600,
                color: "var(--muted)",
                borderTop: "1px solid rgba(255, 255, 255, 0.08)",
                paddingTop: "10px",
                letterSpacing: "0.02em",
              }}
            >
              Yanıt süresi: {formatElapsed(entry.elapsedMs!)}
            </div>
          ) : null}
        </div>
        {showSources ? <SourcesCard chunks={entry.retrievedChunks!} /> : null}
      </div>
    </div>
  );
}

"use client";

/**
 * Single row in the message stream.
 *
 * Layouts
 * -------
 * * `assistant` (default RAG_LLM/EXTRACTIVE): avatar + bubble + footer +
 *   optional <SourcesCard />.
 * * `assistant` (LLM_TIMEOUT, Adım 5.4): avatar + <TimeoutCard /> with a
 *   Retry button instead of a normal bubble.
 * * `assistant` (NO_INFO / FALLBACK, Adım 5.4): normal bubble + 3
 *   <NoInfoSuggestions /> chips so the user has a non-dead-end next step.
 * * `user`: bubble right-aligned, no avatar (matches templates/index.html).
 */
import type { AnswerSource, RetrievedChunk } from "@/app/lib/types";

import NoInfoSuggestions from "./NoInfoSuggestions";
import SourcesCard from "./SourcesCard";
import TimeoutCard from "./TimeoutCard";

export type MessageEntry = {
  /** Stable id is required so React reconciles correctly while streams append. */
  id: string;
  role: "user" | "assistant";
  text: string;
  /** Wall-clock time we waited for the assistant turn (assistant only). */
  elapsedMs?: number;
  /** Sources that the v1 backend reported alongside the assistant answer. */
  retrievedChunks?: RetrievedChunk[];
  /** Adım 5.4: drives which assistant card variant to render. */
  answerSource?: AnswerSource;
};

type Props = {
  entry: MessageEntry;
  /** True only for the LAST timeout entry while a retry call is in flight. */
  retrying?: boolean;
  /** Called by TimeoutCard's Retry button. */
  onRetry?: () => void;
  /** Called when a NoInfoSuggestions chip is picked. */
  onSuggestionPick?: (text: string) => void;
  /** Disables suggestion chips while another /ask is pending. */
  suggestionsDisabled?: boolean;
};

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} sn`;
}

function isNoInfoSource(source: AnswerSource | undefined): boolean {
  return source === "NO_INFO" || source === "FALLBACK";
}

/**
 * Adım 5.5 UX patch
 * -----------------
 * Only treat retrieved_chunks as "real sources to cite" when the answer
 * actually came from RAG. For NO_INFO / FALLBACK / LLM_TIMEOUT the model
 * could not (or did not) ground itself in those chunks, so showing them
 * misleads users — e.g. "Makarna tarifi" returns NO_INFO yet the hybrid
 * retriever still surfaces a Genel Kimya chunk as a top-cosine hit. The
 * answer text says "I don't know"; listing those chunks suggests the
 * opposite.
 *
 * EXTRACTIVE keeps sources because the deterministic extractor takes
 * a sentence directly out of the retrieved blocks.
 */
function shouldShowSources(source: AnswerSource | undefined): boolean {
  return source === "RAG_LLM" || source === "EXTRACTIVE" || source === undefined || source === null;
}

export default function MessageRow({
  entry,
  retrying = false,
  onRetry,
  onSuggestionPick,
  suggestionsDisabled = false,
}: Props) {
  if (entry.role === "user") {
    return (
      <div className="message-row user">
        <div className="msg-main">
          <div className="bubble user">{entry.text}</div>
        </div>
      </div>
    );
  }

  // LLM_TIMEOUT branch — distinct visual + Retry button instead of bubble.
  if (entry.answerSource === "LLM_TIMEOUT") {
    return (
      <div className="message-row assistant">
        <div className="msg-aside">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img className="avatar" src="/avatar.png" alt="ACUdost avatar" aria-hidden="true" />
        </div>
        <div className="msg-main">
          <TimeoutCard retrying={retrying} onRetry={onRetry ?? (() => {})} />
        </div>
      </div>
    );
  }

  const showFooter = typeof entry.elapsedMs === "number";
  const showSources =
    Array.isArray(entry.retrievedChunks) &&
    entry.retrievedChunks.length > 0 &&
    shouldShowSources(entry.answerSource);
  const showSuggestions = isNoInfoSource(entry.answerSource);

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
        {showSuggestions && onSuggestionPick ? (
          <NoInfoSuggestions onPick={onSuggestionPick} disabled={suggestionsDisabled} />
        ) : null}
      </div>
    </div>
  );
}

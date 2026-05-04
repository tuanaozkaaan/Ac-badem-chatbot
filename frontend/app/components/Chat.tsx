"use client";

/**
 * Top-level interactive surface. Owns conversation state and orchestrates
 * the sub-components (Sidebar, TopBanner, TopBar, Welcome, MessageRow,
 * LoadingCard, Composer). Equivalent to what static/.../app.js + chat.js +
 * conversations.js did imperatively in the legacy Django UI.
 *
 * State machine
 * -------------
 *   composing  ── send ──▶ pending  ── upstream success ──▶ composing
 *                              │                              │
 *                              ├── 200 + LLM_TIMEOUT ────────┤  (Retry button on the bubble)
 *                              ├── 200 + NO_INFO/FALLBACK ──┤  (Suggestion chips on the bubble)
 *                              ├── 429 rate_limited ─────────┤  (banner; no extra bubble)
 *                              └── network / 504 ───────────┘  (banner + inline error bubble)
 *
 * `pending` is a single boolean (not a queue) because the legacy UI
 * disabled the input during a request — no concurrent /ask calls. Same
 * here.
 *
 * Adım 5.4 additions
 * ------------------
 * * `lastUserQuestionRef` — captured before every send so a Retry from a
 *   LLM_TIMEOUT card knows what to re-send without scraping the DOM.
 * * `rateLimited`         — short-lived flag (auto-clears after 8s) that
 *   drives <TopBanner mode="rate_limited" />.
 * * `serverUnreachable`   — set on fetch failure / 504, cleared on the
 *   next successful response. Drives <TopBanner mode="server_unreachable" />.
 * * `retryingMessageId`   — id of the LLM_TIMEOUT entry currently being
 *   retried; suppresses the bottom LoadingCard so the inline retry
 *   spinner is the only "in-flight" indicator.
 */
import { useEffect, useRef, useState } from "react";

import { getConversation, listConversations, postAsk } from "@/app/lib/api";
import type {
  AnswerSource,
  ConversationSummary,
  RetrievedChunk,
} from "@/app/lib/types";

import Composer from "./Composer";
import LoadingCard from "./LoadingCard";
import MessageRow, { type MessageEntry } from "./MessageRow";
import Sidebar from "./Sidebar";
import TopBanner from "./TopBanner";
import TopBar from "./TopBar";
import Welcome from "./Welcome";

function makeId(): string {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

const RATE_LIMIT_BANNER_MS = 8_000;

type SendOptions = {
  /** Replace the most recent assistant entry instead of appending one. */
  replaceLastAssistant?: boolean;
  /** Set when the call originated from a TimeoutCard's Retry button. */
  retryingMessageId?: string;
};

export default function Chat() {
  const [messages, setMessages] = useState<MessageEntry[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null);
  const [pending, setPending] = useState<boolean>(false);
  const [composerValue, setComposerValue] = useState<string>("");

  // Adım 5.4 — UX state
  const [rateLimited, setRateLimited] = useState<boolean>(false);
  const [serverUnreachable, setServerUnreachable] = useState<boolean>(false);
  const [retryingMessageId, setRetryingMessageId] = useState<string | null>(null);

  const messagesWrapRef = useRef<HTMLDivElement | null>(null);
  const lastUserQuestionRef = useRef<string | null>(null);
  const rateLimitTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Auto-scroll on every new message OR when the loading card mounts.
  useEffect(() => {
    const el = messagesWrapRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, pending]);

  // Initial conversation list load + reload after every successful /ask.
  // Failures are swallowed: the sidebar is best-effort and must not block the chat.
  const reloadConversations = async () => {
    const r = await listConversations();
    if (r.ok) {
      setConversations(r.data.results || []);
    }
  };

  useEffect(() => {
    void reloadConversations();
    return () => {
      if (rateLimitTimerRef.current) {
        clearTimeout(rateLimitTimerRef.current);
      }
    };
  }, []);

  const triggerRateLimitBanner = () => {
    setRateLimited(true);
    if (rateLimitTimerRef.current) {
      clearTimeout(rateLimitTimerRef.current);
    }
    rateLimitTimerRef.current = setTimeout(() => {
      setRateLimited(false);
      rateLimitTimerRef.current = null;
    }, RATE_LIMIT_BANNER_MS);
  };

  const handleNewChat = () => {
    setCurrentConversationId(null);
    setMessages([]);
    setPending(false);
    setComposerValue("");
    setRetryingMessageId(null);
    lastUserQuestionRef.current = null;
  };

  const handleSelectConversation = async (id: number) => {
    setCurrentConversationId(id);
    setPending(false);
    const r = await getConversation(id);
    if (!r.ok) {
      // 404 means the row no longer belongs to this session (or never did).
      // Treat it like "new chat" so the user is not stuck on a dead id.
      setMessages([]);
      setCurrentConversationId(null);
      return;
    }
    setServerUnreachable(false);
    const entries: MessageEntry[] = (r.data.messages || []).map((m) => ({
      id: `${m.id}`,
      role: m.role,
      text: m.content,
    }));
    setMessages(entries);
  };

  /**
   * Replace OR append an assistant entry depending on the send mode.
   * Wrapped in a helper so the success path and the various error paths
   * share identical reconciliation logic.
   */
  const upsertAssistantEntry = (entry: MessageEntry, replace: boolean) => {
    setMessages((prev) => {
      if (!replace) return [...prev, entry];
      const next = [...prev];
      for (let i = next.length - 1; i >= 0; i--) {
        if (next[i].role === "assistant") {
          next[i] = entry;
          return next;
        }
      }
      next.push(entry);
      return next;
    });
  };

  /**
   * Single source of truth for "send a question" — composer Submit, welcome
   * suggestions, NO_INFO chips, and the LLM_TIMEOUT Retry button all funnel
   * through here so the in-flight invariants (pending, conversation id,
   * banner state) cannot diverge between code paths.
   */
  const sendQuestion = async (question: string, opts: SendOptions = {}) => {
    const trimmed = question.trim();
    if (!trimmed || pending) return;

    lastUserQuestionRef.current = trimmed;

    if (!opts.replaceLastAssistant) {
      const userEntry: MessageEntry = { id: makeId(), role: "user", text: trimmed };
      setMessages((prev) => [...prev, userEntry]);
      setComposerValue("");
    }

    setPending(true);
    if (opts.retryingMessageId) {
      setRetryingMessageId(opts.retryingMessageId);
    }

    const t0 = performance.now();
    try {
      const result = await postAsk({
        question: trimmed,
        conversation_id: currentConversationId,
      });
      const elapsedMs = performance.now() - t0;

      if (!result.ok) {
        if (typeof result.error.conversation_id === "number") {
          setCurrentConversationId(result.error.conversation_id);
        }

        const code = result.error.error?.code;
        if (code === "rate_limited") {
          // Banner + leave the user message in place; no error bubble noise.
          triggerRateLimitBanner();
          return;
        }
        if (code === "upstream_unreachable" || result.status >= 500) {
          setServerUnreachable(true);
        }

        const errorMsg = result.error.error?.message || `HTTP ${result.status}`;
        upsertAssistantEntry(
          {
            id: makeId(),
            role: "assistant",
            text: `Bağlantı hatası: ${errorMsg}`,
            elapsedMs,
            answerSource: "LLM_TIMEOUT",
            // We deliberately reuse LLM_TIMEOUT for upstream_unreachable too
            // so the user gets the same Retry affordance — fixing flaky
            // network conditions usually only requires re-sending.
          },
          opts.replaceLastAssistant === true,
        );
        return;
      }

      // Success — clear any leftover "server is dead" banner.
      setServerUnreachable(false);

      const { data } = result;
      if (typeof data.conversation_id === "number") {
        setCurrentConversationId(data.conversation_id);
      }
      const chunks: RetrievedChunk[] = data.retrieved_chunks || [];
      const source: AnswerSource = data.answer_source ?? null;

      upsertAssistantEntry(
        {
          id: makeId(),
          role: "assistant",
          text: data.answer || "Yanıt boş döndü.",
          elapsedMs,
          retrievedChunks: chunks,
          answerSource: source,
        },
        opts.replaceLastAssistant === true,
      );
    } catch (error) {
      // postAsk surfaces AbortError through the result; anything raised here
      // is genuinely unexpected (e.g. network TypeError before we even got
      // a response). Show the banner AND a contextual error bubble.
      setServerUnreachable(true);
      const elapsedMs = performance.now() - t0;
      const message = error instanceof Error ? error.message : "Bilinmeyen hata";
      upsertAssistantEntry(
        {
          id: makeId(),
          role: "assistant",
          text: `Bağlantı hatası: ${message}`,
          elapsedMs,
          answerSource: "LLM_TIMEOUT",
        },
        opts.replaceLastAssistant === true,
      );
    } finally {
      setPending(false);
      setRetryingMessageId(null);
      void reloadConversations();
    }
  };

  const handleSubmit = () => {
    void sendQuestion(composerValue);
  };

  const handleSuggestionClick = (text: string) => {
    if (pending) return;
    void sendQuestion(text);
  };

  const handleRetry = (messageId: string) => {
    if (pending) return;
    const last = lastUserQuestionRef.current;
    if (!last) return;
    void sendQuestion(last, { replaceLastAssistant: true, retryingMessageId: messageId });
  };

  // Hide the bottom LoadingCard when a retry is in flight; the inline
  // TimeoutCard already shows its own spinner state.
  const showLoadingCard = pending && !retryingMessageId;

  return (
    <>
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
      />
      <main className="main">
        <TopBanner rateLimited={rateLimited} serverUnreachable={serverUnreachable} />
        <TopBar />
        <section className="chat-shell">
          <div className="messages-wrap" ref={messagesWrapRef}>
            <div className="messages">
              {messages.map((m) => (
                <MessageRow
                  key={m.id}
                  entry={m}
                  retrying={retryingMessageId === m.id}
                  onRetry={() => handleRetry(m.id)}
                  onSuggestionPick={handleSuggestionClick}
                  suggestionsDisabled={pending}
                />
              ))}
              {showLoadingCard ? <LoadingCard /> : null}
            </div>
            <Welcome
              visible={messages.length === 0 && !pending}
              onSuggestionClick={handleSuggestionClick}
            />
          </div>
          <Composer
            value={composerValue}
            disabled={pending}
            onChange={setComposerValue}
            onSubmit={handleSubmit}
          />
        </section>
      </main>
    </>
  );
}

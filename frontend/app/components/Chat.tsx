"use client";

/**
 * Top-level interactive surface. Owns conversation state and orchestrates
 * the sub-components (Sidebar, TopBar, Welcome, MessageRow, LoadingCard,
 * Composer). Equivalent to what static/.../app.js + chat.js +
 * conversations.js did imperatively in the legacy Django UI.
 *
 * State machine (rough)
 * ---------------------
 *   composing  ── send ──▶ pending  ── upstream success ──▶ composing
 *                              │
 *                              └── error / timeout ─▶ composing
 *
 * `pending` is a single boolean (not a queue) because the legacy UI
 * disabled the input during a request — no concurrent /ask calls. Same
 * here.
 */
import { useEffect, useRef, useState } from "react";

import {
  getConversation,
  listConversations,
  postAsk,
} from "@/app/lib/api";
import type {
  ConversationSummary,
  RetrievedChunk,
} from "@/app/lib/types";

import Composer from "./Composer";
import LoadingCard from "./LoadingCard";
import MessageRow, { type MessageEntry } from "./MessageRow";
import Sidebar from "./Sidebar";
import TopBar from "./TopBar";
import Welcome from "./Welcome";

function makeId(): string {
  // Random + timestamp is sufficient for React keys; we are not persisting these.
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function Chat() {
  const [messages, setMessages] = useState<MessageEntry[]>([]);
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null);
  const [pending, setPending] = useState<boolean>(false);
  const [composerValue, setComposerValue] = useState<string>("");

  const messagesWrapRef = useRef<HTMLDivElement | null>(null);

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
  }, []);

  const handleNewChat = () => {
    setCurrentConversationId(null);
    setMessages([]);
    setPending(false);
    setComposerValue("");
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
    const entries: MessageEntry[] = (r.data.messages || []).map((m) => ({
      id: `${m.id}`,
      role: m.role,
      text: m.content,
    }));
    setMessages(entries);
  };

  /**
   * Single source of truth for "send a question" — both the composer Submit
   * handler and the welcome-card suggestion buttons funnel through here so
   * we can never split the in-flight invariants between two code paths.
   */
  const sendQuestion = async (question: string) => {
    const trimmed = question.trim();
    if (!trimmed || pending) return;

    const userEntry: MessageEntry = { id: makeId(), role: "user", text: trimmed };
    setMessages((prev) => [...prev, userEntry]);
    setComposerValue("");
    setPending(true);

    const t0 = performance.now();
    try {
      const result = await postAsk({
        question: trimmed,
        conversation_id: currentConversationId,
      });
      const elapsedMs = performance.now() - t0;

      if (!result.ok) {
        // The backend may attach a `conversation_id` even on failure (e.g. an
        // LLM error mid-turn): keep it so retries land on the same row.
        if (typeof result.error.conversation_id === "number") {
          setCurrentConversationId(result.error.conversation_id);
        }
        const errorMsg = result.error.error?.message || `HTTP ${result.status}`;
        setMessages((prev) => [
          ...prev,
          { id: makeId(), role: "assistant", text: `API hatası: ${errorMsg}`, elapsedMs },
        ]);
        return;
      }

      const { data } = result;
      if (typeof data.conversation_id === "number") {
        setCurrentConversationId(data.conversation_id);
      }
      const chunks: RetrievedChunk[] = data.retrieved_chunks || [];
      setMessages((prev) => [
        ...prev,
        {
          id: makeId(),
          role: "assistant",
          text: data.answer || "Yanıt boş döndü.",
          elapsedMs,
          retrievedChunks: chunks,
        },
      ]);
    } catch (error) {
      // postAsk handles AbortError by surfacing it through the result, so
      // anything caught here is genuinely unexpected. Keep it visible to the
      // user instead of silently dropping.
      const elapsedMs = performance.now() - t0;
      const message = error instanceof Error ? error.message : "Bilinmeyen hata";
      setMessages((prev) => [
        ...prev,
        { id: makeId(), role: "assistant", text: `Bağlantı hatası: ${message}`, elapsedMs },
      ]);
    } finally {
      setPending(false);
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

  return (
    <>
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelect={handleSelectConversation}
        onNewChat={handleNewChat}
      />
      <main className="main">
        <TopBar />
        <section className="chat-shell">
          <div className="messages-wrap" ref={messagesWrapRef}>
            <div className="messages">
              {messages.map((m) => (
                <MessageRow key={m.id} entry={m} />
              ))}
              {pending ? <LoadingCard /> : null}
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

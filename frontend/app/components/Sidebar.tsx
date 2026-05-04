"use client";

/**
 * Left-hand conversations panel. Mirrors the .sidebar block in
 * templates/index.html and the click handlers in static/.../app.js +
 * conversations.js, but driven by React state rather than direct DOM
 * mutation.
 */
import type { ConversationSummary } from "@/app/lib/types";

type Props = {
  conversations: ConversationSummary[];
  currentConversationId: number | null;
  onSelect: (id: number) => void;
  onNewChat: () => void;
};

function formatShortTime(iso: string): string {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleString("tr-TR", {
      day: "2-digit",
      month: "short",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch (_e) {
    return "";
  }
}

function formatConvTitle(c: ConversationSummary): string {
  const t = c?.title ? String(c.title).trim() : "";
  return t || "Yeni sohbet";
}

export default function Sidebar({
  conversations,
  currentConversationId,
  onSelect,
  onNewChat,
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img className="sidebar-brand-avatar" src="/avatar.png" alt="ACUdost avatar" />
        <div className="sidebar-brand-text">
          <strong>ACUdost</strong>
          <span>Akademik bilgi asistanı</span>
        </div>
      </div>
      <button type="button" className="new-chat-btn" onClick={onNewChat}>
        <svg
          width="18"
          height="18"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          aria-hidden="true"
        >
          <path d="M12 5v14M5 12h14" strokeLinecap="round" />
        </svg>
        Yeni sohbet
      </button>
      <div className="sidebar-heading">Geçmiş</div>
      <div className="conv-list" aria-label="Sohbet geçmişi">
        {conversations.map((c) => {
          const isActive = c.id === currentConversationId;
          return (
            <button
              key={c.id}
              type="button"
              className={`conv-item${isActive ? " active" : ""}`}
              data-id={c.id}
              onClick={() => onSelect(c.id)}
            >
              <div className="conv-item-title">{formatConvTitle(c)}</div>
              <div className="conv-item-meta">{formatShortTime(c.updated_at)}</div>
            </button>
          );
        })}
      </div>
    </aside>
  );
}

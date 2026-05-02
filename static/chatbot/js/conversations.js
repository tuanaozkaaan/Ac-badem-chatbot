/* conversations.js — sidebar history list + select handlers.
 *
 * Depends on api.js (fetchJSON, API_CONV) and chat.js (clearLoading,
 * appendStaticBubble, renderWelcome, scrollToBottom, conversation array).
 */

let conversationsCache = [];

function formatConvTitle(c) {
  const t = c && c.title ? String(c.title).trim() : "";
  return t || "Yeni sohbet";
}

function formatShortTime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    return d.toLocaleString("tr-TR", { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });
  } catch (_e) {
    return "";
  }
}

async function loadConversations() {
  if (!convListEl) return;
  try {
    const r = await fetchJSON(API_CONV);
    if (!r.ok) return;
    const data = await r.json();
    conversationsCache = Array.isArray(data.results) ? data.results : [];
    renderConvList();
  } catch (_e) {
    // intentionally swallow — sidebar load failures must not break the chat
  }
}

function renderConvList() {
  if (!convListEl) return;
  convListEl.innerHTML = "";
  for (const c of conversationsCache) {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "conv-item" + (c.id === currentConversationId ? " active" : "");
    btn.dataset.id = String(c.id);
    const title = document.createElement("div");
    title.className = "conv-item-title";
    title.textContent = formatConvTitle(c);
    const meta = document.createElement("div");
    meta.className = "conv-item-meta";
    meta.textContent = formatShortTime(c.updated_at);
    btn.appendChild(title);
    btn.appendChild(meta);
    btn.addEventListener("click", () => {
      void selectConversation(c.id);
    });
    convListEl.appendChild(btn);
  }
}

async function selectConversation(id) {
  currentConversationId = id;
  clearLoading();
  try {
    const r = await fetchJSON(`${API_CONV}${id}/`);
    if (!r.ok) return;
    const data = await r.json();
    messagesEl.innerHTML = "";
    conversation.length = 0;
    const msgs = Array.isArray(data.messages) ? data.messages : [];
    for (const m of msgs) {
      conversation.push({ role: m.role, content: m.content });
      appendStaticBubble(m.role, m.content);
    }
    renderWelcome();
    renderConvList();
    scrollToBottom();
  } catch (_e) {
    // intentionally swallow — selection failures must not break the chat
  }
}

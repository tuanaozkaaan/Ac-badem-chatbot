/* chat.js — message rendering, loading state, and the /ask round-trip.
 *
 * Globals defined here are consumed by conversations.js (rendering helpers) and
 * app.js (event wiring). No DOM lookups happen at script load time; the helpers
 * defer to globals (messagesEl, etc.) that app.js fills in on boot.
 */

const conversation = [];
let loadingBubble = null;
let currentConversationId = null;

function mountMessageRow(role, bubble) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;
  if (role === "assistant") {
    const aside = document.createElement("div");
    aside.className = "msg-aside";
    const av = document.createElement("img");
    av.className = "avatar";
    av.src = ASSISTANT_AVATAR_URL;
    av.alt = "ACUdost avatar";
    av.setAttribute("aria-hidden", "true");
    aside.appendChild(av);
    row.appendChild(aside);
  }
  const main = document.createElement("div");
  main.className = "msg-main";
  main.appendChild(bubble);
  row.appendChild(main);
  messagesEl.appendChild(row);
  return row;
}

function appendStaticBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  mountMessageRow(role, bubble);
}

function autoResizeInput() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 200)}px`;
}

function scrollToBottom() {
  messagesWrapEl.scrollTop = messagesWrapEl.scrollHeight;
}

function renderWelcome() {
  welcomeEl.style.display = conversation.length ? "none" : "block";
}

function createMessageBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;
  bubble.textContent = text;
  mountMessageRow(role, bubble);
  return bubble;
}

function addMessage(role, text) {
  conversation.push({ role, text });
  createMessageBubble(role, text);
  renderWelcome();
  scrollToBottom();
}

function formatElapsed(ms) {
  if (ms < 1000) return `${Math.round(ms)} ms`;
  return `${(ms / 1000).toFixed(2)} sn`;
}

/** Assistant bubble + footer with round-trip time for /ask POST */
function addAssistantMessageWithTiming(text, elapsedMs) {
  conversation.push({ role: "assistant", text, elapsedMs });
  const bubble = document.createElement("div");
  bubble.className = "bubble assistant";
  const body = document.createElement("div");
  body.style.whiteSpace = "pre-wrap";
  body.style.wordBreak = "break-word";
  body.style.lineHeight = "1.55";
  body.style.fontSize = "15px";
  body.textContent = text;
  const foot = document.createElement("div");
  foot.style.marginTop = "12px";
  foot.style.fontSize = "11px";
  foot.style.fontWeight = "600";
  foot.style.color = "var(--muted)";
  foot.style.borderTop = "1px solid rgba(255, 255, 255, 0.08)";
  foot.style.paddingTop = "10px";
  foot.style.letterSpacing = "0.02em";
  foot.textContent = `Yanıt süresi: ${formatElapsed(elapsedMs)}`;
  bubble.appendChild(body);
  bubble.appendChild(foot);
  mountMessageRow("assistant", bubble);
  renderWelcome();
  scrollToBottom();
}

function showLoading() {
  const row = document.createElement("div");
  row.className = "message-row assistant";
  const aside = document.createElement("div");
  aside.className = "msg-aside";
  const av = document.createElement("img");
  av.className = "avatar";
  av.src = ASSISTANT_AVATAR_URL;
  av.alt = "ACUdost avatar";
  av.setAttribute("aria-hidden", "true");
  aside.appendChild(av);
  const main = document.createElement("div");
  main.className = "msg-main";
  const card = document.createElement("div");
  card.className = "loading-card";
  card.innerHTML =
    '<div style="font-size:14px;font-weight:700;color:var(--text);letter-spacing:-0.02em">Yanıt hazırlanıyor</div>' +
    '<div style="font-size:12px;color:var(--muted);margin-top:6px;line-height:1.5;font-weight:500">Veri tabanından ilgili metinler seçiliyor; model yanıtı üretiyor. Yoğun yüklemede bu adım birkaç dakika sürebilir.</div>' +
    '<div class="loading-shimmer"></div><div class="loading-shimmer narrow"></div>';
  main.appendChild(card);
  row.appendChild(aside);
  row.appendChild(main);
  messagesEl.appendChild(row);
  loadingBubble = row;
  renderWelcome();
  scrollToBottom();
}

function clearLoading() {
  if (loadingBubble) {
    loadingBubble.remove();
    loadingBubble = null;
  }
}

function setSendingState(isSending) {
  sendBtn.disabled = isSending;
  questionInput.disabled = isSending;
}

async function sendMessage() {
  const question = questionInput.value.trim();
  if (!question) return;

  addMessage("user", question);
  questionInput.value = "";
  autoResizeInput();
  setSendingState(true);
  showLoading();

  const t0 = performance.now();
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  const payload = { question };
  if (currentConversationId != null) {
    payload.conversation_id = currentConversationId;
  }
  try {
    const response = await fetchJSON(API_URL, {
      method: "POST",
      body: payload,
      signal: controller.signal,
    });

    clearLoading();

    if (!response.ok) {
      let errMessage = `HTTP ${response.status}`;
      const rawError = await response.text();
      try {
        const errJson = JSON.parse(rawError || "{}");
        errMessage = errJson.detail || errJson.error || errMessage;
        if (typeof errJson.conversation_id === "number") {
          currentConversationId = errJson.conversation_id;
        }
      } catch (_e) {
        errMessage = (rawError || errMessage).slice(0, 300);
      }
      addAssistantMessageWithTiming(`API error: ${errMessage}`, performance.now() - t0);
      void loadConversations();
      return;
    }

    const data = await response.json();
    if (typeof data.conversation_id === "number") {
      currentConversationId = data.conversation_id;
    }
    addAssistantMessageWithTiming(data.answer || "No answer returned.", performance.now() - t0);
    void loadConversations();
  } catch (error) {
    clearLoading();
    const elapsedMs = performance.now() - t0;
    if (error && error.name === "AbortError") {
      addAssistantMessageWithTiming(
        "İstek zaman aşımı (7 dk): sunucu yanıt vermedi. Ollama/model yavaş olabilir; embedding ilk yüklemede uzun sürebilir. Bir süre sonra tekrar dene veya web/ollama konteynerlarını kontrol et.",
        elapsedMs
      );
    } else {
      addAssistantMessageWithTiming(`Connection error: ${error.message}`, elapsedMs);
    }
    void loadConversations();
  } finally {
    window.clearTimeout(timeoutId);
    setSendingState(false);
    questionInput.focus();
  }
}

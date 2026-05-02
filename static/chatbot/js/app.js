/* app.js — DOM references, suggestion cards, event wiring, and boot.
 *
 * This is the last script loaded; it pulls together symbols defined in
 * api.js / chat.js / conversations.js and starts the application.
 */

const messagesEl = document.getElementById("messages");
const messagesWrapEl = document.getElementById("messagesWrap");
const welcomeEl = document.getElementById("welcome");
const suggestionsEl = document.getElementById("suggestions");
const questionInput = document.getElementById("questionInput");
const sendBtn = document.getElementById("sendBtn");
const newChatBtn = document.getElementById("newChatBtn");
const convListEl = document.getElementById("convList");

const SUGGESTIONS = [
  { label: "Genel", text: "Acıbadem Üniversitesi hakkında kısa bilgi verir misiniz?" },
  { label: "Kampüs", text: "Acıbadem Üniversitesi kampüs adresi ve ulaşım bilgisi nedir?" },
  { label: "Akademik", text: "Mühendislik ve Doğa Bilimleri Fakültesi hangi bölümleri içerir?" },
  { label: "İletişim", text: "Öğrenci işleri ve genel iletişim telefon/e-posta bilgileri nelerdir?" },
];

function renderSuggestions() {
  if (!suggestionsEl) return;
  suggestionsEl.innerHTML = "";
  for (const item of SUGGESTIONS) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "suggestion-card";
    card.innerHTML = `<span class="label">${item.label}</span><span class="text">${item.text}</span>`;
    card.addEventListener("click", () => {
      // Fill + send immediately (ChatGPT-like quick action)
      questionInput.value = item.text;
      autoResizeInput();
      sendMessage();
    });
    suggestionsEl.appendChild(card);
  }
}

sendBtn.addEventListener("click", sendMessage);
newChatBtn.addEventListener("click", () => {
  currentConversationId = null;
  conversation.length = 0;
  messagesEl.innerHTML = "";
  clearLoading();
  renderWelcome();
  renderConvList();
  questionInput.focus();
});

questionInput.addEventListener("input", autoResizeInput);
questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

renderWelcome();
renderSuggestions();
autoResizeInput();
questionInput.focus();
void loadConversations();

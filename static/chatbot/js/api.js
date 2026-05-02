/* api.js — endpoint constants, CSRF cookie reader, fetch wrapper.
 *
 * Loaded first; later scripts (chat.js, conversations.js, app.js) consume the
 * symbols defined here. Keep this file dependency-free so the JS bundle has
 * a clean root.
 */

// Same-origin; versioned paths match ``path("api/v1/", include("chatbot.urls"))``.
const API_URL = "/api/v1/ask";
const API_CONV = "/api/v1/conversations/";

/* Embedding matrisi ilk yüklemede + Ollama yavaşsa 4 dk yetmez; 7 dk. */
const FETCH_TIMEOUT_MS = 420000;

// Avatar URL is injected by Django via <body data-avatar-url="..."> so this file
// can be served as a static asset without going through the template engine.
const ASSISTANT_AVATAR_URL = (document.body && document.body.dataset.avatarUrl) || "";

/** Read a cookie value by name, returning null when absent. */
function getCookie(name) {
  const cookies = document.cookie ? document.cookie.split(";") : [];
  for (const raw of cookies) {
    const [k, ...rest] = raw.trim().split("=");
    if (k === name) return decodeURIComponent((rest.length ? rest.join("=") : "") || "");
  }
  return null;
}

/**
 * CSRF value for X-CSRFToken: prefer server-injected body attribute (always in
 * sync with the cookie set by ensure_csrf_cookie), fall back to reading csrftoken.
 */
function getCsrfToken() {
  const el = document.body;
  const fromDom = el && el.dataset && el.dataset.csrfToken ? String(el.dataset.csrfToken).trim() : "";
  if (fromDom) return fromDom;
  const fromCookie = getCookie("csrftoken");
  return fromCookie ? String(fromCookie).trim() : "";
}

/**
 * fetch() wrapper that:
 *   - serializes JSON request bodies,
 *   - injects X-CSRFToken on mutating requests (POST/PUT/PATCH/DELETE),
 *   - leaves GET/HEAD requests untouched (Django CSRF middleware exempts them).
 *
 * Returns the raw Response so callers keep full control over body parsing.
 */
async function fetchJSON(url, { method = "GET", body, headers, signal } = {}) {
  const headerMap = { ...(headers || {}) };
  const init = {
    method,
    headers: headerMap,
    signal,
    credentials: "same-origin",
  };
  if (body !== undefined) {
    init.body = typeof body === "string" ? body : JSON.stringify(body);
    if (!headerMap["Content-Type"] && !headerMap["content-type"]) {
      headerMap["Content-Type"] = "application/json";
    }
  }
  const upper = String(method).toUpperCase();
  if (upper !== "GET" && upper !== "HEAD") {
    const csrf = getCsrfToken();
    if (csrf) headerMap["X-CSRFToken"] = csrf;
  }
  return fetch(url, init);
}

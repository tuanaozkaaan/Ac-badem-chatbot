/* api.js — endpoint constants, CSRF cookie reader, fetch wrapper.
 *
 * Loaded first; later scripts (chat.js, conversations.js, app.js) consume the
 * symbols defined here. Keep this file dependency-free so the JS bundle has
 * a clean root.
 */

// Same-origin by default so it works with both runserver and Docker port mappings.
const API_URL = "/ask";
const API_CONV = "/api/conversations/";

/* Embedding matrisi ilk yüklemede + Ollama yavaşsa 4 dk yetmez; 7 dk. */
const FETCH_TIMEOUT_MS = 420000;

// Avatar URL is injected by Django via <body data-avatar-url="..."> so this file
// can be served as a static asset without going through the template engine.
const ASSISTANT_AVATAR_URL = (document.body && document.body.dataset.avatarUrl) || "";

/** Read a cookie value by name, returning null when absent. */
function getCookie(name) {
  const cookies = document.cookie ? document.cookie.split(";") : [];
  for (const raw of cookies) {
    const [k, v] = raw.trim().split("=");
    if (k === name) return decodeURIComponent(v || "");
  }
  return null;
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
  const init = { method, headers: { ...(headers || {}) }, signal };
  if (body !== undefined) {
    init.body = typeof body === "string" ? body : JSON.stringify(body);
    if (!init.headers["Content-Type"]) {
      init.headers["Content-Type"] = "application/json";
    }
  }
  const upper = String(method).toUpperCase();
  if (upper !== "GET" && upper !== "HEAD") {
    const csrf = getCookie("csrftoken");
    if (csrf) init.headers["X-CSRFToken"] = csrf;
  }
  return fetch(url, init);
}

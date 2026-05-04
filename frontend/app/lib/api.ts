/**
 * Browser-side fetch helpers. Every URL here is same-origin
 * (``/api/...``) so the call hits the Next.js Route Handler proxies
 * defined in app/api/*. The browser must NEVER address Django directly;
 * keeping all fetch calls relative makes that invariant impossible to
 * accidentally break.
 *
 * The legacy chat.js used a 7-minute timeout because the local Gemma
 * model can be cold; we keep that envelope here for parity.
 */
import type {
  ApiError,
  AskRequest,
  AskResponse,
  ConversationDetail,
  ConversationsListResponse,
} from "./types";

const ASK_URL = "/api/ask";
const CONVERSATIONS_URL = "/api/conversations";
const FETCH_TIMEOUT_MS = 420_000;

/** Result discriminated union — callers branch on ``ok`` instead of try/catch. */
export type FetchResult<T> =
  | { ok: true; data: T; status: number }
  | { ok: false; error: ApiError; status: number };

async function readJson<T>(response: Response): Promise<T | ApiError> {
  const text = await response.text();
  if (!text) {
    return { error: { code: "internal_error", message: "Empty response body." } } as ApiError;
  }
  try {
    return JSON.parse(text) as T | ApiError;
  } catch (_e) {
    return {
      error: { code: "internal_error", message: text.slice(0, 300) || "Unparseable response body." },
    } as ApiError;
  }
}

function asResult<T>(response: Response, parsed: T | ApiError): FetchResult<T> {
  if (response.ok && parsed && typeof parsed === "object" && !("error" in parsed)) {
    return { ok: true, data: parsed as T, status: response.status };
  }
  // Failure cases: either non-2xx status, or backend emitted an error envelope
  // even on 200 (defensive). Normalize to the same FetchResult shape.
  if (parsed && typeof parsed === "object" && "error" in parsed) {
    return { ok: false, error: parsed as ApiError, status: response.status };
  }
  return {
    ok: false,
    error: { error: { code: "internal_error", message: `HTTP ${response.status}` } },
    status: response.status,
  };
}

export async function postAsk(
  payload: AskRequest,
  signal?: AbortSignal,
): Promise<FetchResult<AskResponse>> {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  // Compose with caller-provided signal so an outer AbortController can still cancel.
  signal?.addEventListener("abort", () => controller.abort(), { once: true });
  try {
    const response = await fetch(ASK_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
      credentials: "same-origin",
    });
    const parsed = await readJson<AskResponse>(response);
    return asResult<AskResponse>(response, parsed);
  } finally {
    window.clearTimeout(timeoutId);
  }
}

export async function listConversations(): Promise<FetchResult<ConversationsListResponse>> {
  const response = await fetch(CONVERSATIONS_URL, {
    method: "GET",
    credentials: "same-origin",
  });
  const parsed = await readJson<ConversationsListResponse>(response);
  return asResult<ConversationsListResponse>(response, parsed);
}

export async function getConversation(id: number): Promise<FetchResult<ConversationDetail>> {
  const response = await fetch(`${CONVERSATIONS_URL}/${id}`, {
    method: "GET",
    credentials: "same-origin",
  });
  const parsed = await readJson<ConversationDetail>(response);
  return asResult<ConversationDetail>(response, parsed);
}

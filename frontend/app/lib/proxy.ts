/**
 * Server-side helpers that forward incoming Next.js requests to the Django
 * backend at ``ACU_BACKEND_URL`` (Adım 5.2 proxy strategy).
 *
 * Why a proxy at all
 * ------------------
 * The chatbot is split between two origins (the Next.js host the browser
 * talks to, and Django which serves /api/v1/*). Hitting Django directly
 * from the browser would force CORS + CSRF acrobatics. Routing through
 * Next means:
 *   - the browser sees a single origin → no CORS preflight,
 *   - cookies flow through unchanged so Django's session-bound conversation
 *     ownership (IDOR protection) keeps working,
 *   - secrets like ``ACU_BACKEND_URL`` never leak to the client.
 *
 * Cookie / header rules
 * ---------------------
 * * Only forward end-to-end browser-controlled headers; ``host`` /
 *   ``content-length`` / ``connection`` MUST be re-derived by ``fetch`` itself.
 * * Set-Cookie returned by Django is mirrored verbatim onto the response we
 *   emit to the browser, so the Django sessionid cookie ends up on
 *   ``acudost.example.com`` (the Next.js origin) instead of leaking the
 *   Django origin to the user agent.
 */
import { NextRequest, NextResponse } from "next/server";

import type { ApiError } from "./types";

/**
 * Trim trailing slash so callers can append a leading-slashed path safely.
 * Throws if the env variable is missing because requests without a backend
 * have no chance of succeeding — better to fail loudly than to surface a
 * confusing 500 to the user.
 */
export function getBackendUrl(): string {
  const raw = process.env.ACU_BACKEND_URL;
  if (!raw || !raw.trim()) {
    throw new Error(
      "ACU_BACKEND_URL is not set. Copy frontend/.env.local.example to .env.local " +
        "or export ACU_BACKEND_URL in the runtime env.",
    );
  }
  return raw.replace(/\/+$/, "");
}

/**
 * Whitelist of incoming request headers we forward to Django. Excluded:
 *   - host / connection / content-length: managed by `fetch`,
 *   - x-forwarded-* : Django's SECURE_PROXY_SSL_HEADER would honour stale values.
 */
const FORWARD_HEADERS = new Set([
  "accept",
  "accept-language",
  "content-type",
  "cookie",
  "user-agent",
]);

function buildForwardedHeaders(req: NextRequest): Headers {
  const headers = new Headers();
  req.headers.forEach((value, key) => {
    if (FORWARD_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  });
  return headers;
}

/** Mirror Set-Cookie + Content-Type back to the browser. */
function copyResponseHeaders(upstream: Response, target: Headers): void {
  // Preserve every Set-Cookie individually; Headers#append (not #set) is required
  // because the upstream response may include multiple values.
  upstream.headers.forEach((value, key) => {
    const k = key.toLowerCase();
    if (k === "content-type" || k === "set-cookie" || k === "cache-control") {
      target.append(key, value);
    }
  });
}

export type ProxyOptions = {
  /** Path on the Django backend, must start with ``/api/v1/``. */
  path: string;
  /** Forward query string from the incoming request. Default true. */
  forwardQuery?: boolean;
  /**
   * Override timeout for the upstream call. The default is generous because
   * /api/v1/ask can take several minutes when Ollama is cold.
   */
  timeoutMs?: number;
};

const DEFAULT_TIMEOUT_MS = 7 * 60 * 1000; // 7 minutes — matches legacy chat.js.

/**
 * Forward an incoming Next.js request to Django and stream the JSON response
 * back to the browser. On network failure (Django down, DNS error, timeout)
 * emits the canonical v1 error envelope so the client UI can render a
 * uniform error state.
 */
export async function proxyToBackend(
  req: NextRequest,
  opts: ProxyOptions,
): Promise<NextResponse> {
  const backend = getBackendUrl();
  const url = new URL(opts.path, backend + "/");
  if (opts.forwardQuery !== false) {
    req.nextUrl.searchParams.forEach((value, key) => {
      url.searchParams.set(key, value);
    });
  }

  const init: RequestInit = {
    method: req.method,
    headers: buildForwardedHeaders(req),
    redirect: "manual",
  };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), opts.timeoutMs ?? DEFAULT_TIMEOUT_MS);
  init.signal = controller.signal;

  let upstream: Response;
  try {
    upstream = await fetch(url, init);
  } catch (err) {
    clearTimeout(timeout);
    const isAbort = err instanceof Error && err.name === "AbortError";
    const body: ApiError = {
      error: {
        code: "upstream_unreachable",
        message: isAbort
          ? "Backend did not respond in time."
          : `Backend unreachable: ${err instanceof Error ? err.message : String(err)}`,
      },
    };
    return NextResponse.json(body, { status: 504 });
  }
  clearTimeout(timeout);

  const text = await upstream.text();
  const headers = new Headers();
  copyResponseHeaders(upstream, headers);
  return new NextResponse(text, {
    status: upstream.status,
    headers,
  });
}

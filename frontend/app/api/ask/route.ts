/**
 * POST /api/ask  →  Django POST /api/v1/ask  (Adım 5.1 contract).
 *
 * The browser never talks to Django directly; this route is the entire
 * contract surface from the React side. The handler is intentionally
 * dumb: it forwards the body verbatim and trusts ``proxyToBackend`` to
 * preserve cookies + Set-Cookie + status codes. Body shape is
 * ``AskRequest`` and the upstream returns ``AskResponse`` or ``ApiError``.
 */
import type { NextRequest } from "next/server";

import { proxyToBackend } from "@/app/lib/proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
  return proxyToBackend(req, { path: "/api/v1/ask" });
}

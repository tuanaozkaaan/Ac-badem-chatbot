/**
 * GET  /api/conversations   →  GET  /api/v1/conversations/
 * POST /api/conversations   →  POST /api/v1/conversations/
 *
 * GET returns the list owned by the current Django session (the cookie
 * forwarded by ``proxyToBackend``). POST is included for completeness even
 * though the current UI never calls it directly — ``run_ask`` lazily
 * creates a Conversation row on the first /ask of a thread.
 */
import type { NextRequest } from "next/server";

import { proxyToBackend } from "@/app/lib/proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  return proxyToBackend(req, { path: "/api/v1/conversations/" });
}

export async function POST(req: NextRequest) {
  return proxyToBackend(req, { path: "/api/v1/conversations/" });
}

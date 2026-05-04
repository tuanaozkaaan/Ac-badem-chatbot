/**
 * GET /api/conversations/[id]  →  GET /api/v1/conversations/{id}/
 *
 * Re-uses ``proxyToBackend`` so cookie forwarding, error envelope, and the
 * 7-minute timeout are inherited unchanged. Django returns 404 (not 403)
 * when ``[id]`` belongs to another browser session — that 404 flows to
 * the client untouched, preserving the IDOR guarantee end-to-end.
 */
import type { NextRequest } from "next/server";

import { proxyToBackend } from "@/app/lib/proxy";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

type Params = { params: { id: string } };

export async function GET(req: NextRequest, { params }: Params) {
  // Defensive: only digits go on the wire so a stray "/api/v1/conversations/foo/"
  // never reaches Django and we return a clean 400 here instead of leaking the
  // upstream's URL parser error.
  const idStr = String(params?.id ?? "");
  if (!/^\d+$/.test(idStr)) {
    return Response.json(
      { error: { code: "invalid_conversation_id", message: "Conversation id must be a positive integer." } },
      { status: 400 },
    );
  }
  return proxyToBackend(req, { path: `/api/v1/conversations/${idStr}/` });
}

# ACUdost — Next.js frontend

Next.js 14 (App Router) port of the legacy Django UI. Visually identical to
`templates/index.html` — design tokens, fonts, layout, and JS behaviours
were copied 1:1; nothing was redesigned.

The browser only ever talks to this Next.js process; all chatbot calls go
through Route Handlers under `app/api/*`, which proxy server-side to the
Django backend defined by `ACU_BACKEND_URL` (Adım 5.2 strategy).

## Prerequisites

- Node.js ≥ 18.17 (tested on 22.x)
- A running Django backend that exposes `/api/v1/*` (Adım 5.1).

## First run

```powershell
cd frontend
npm install
copy .env.local.example .env.local      # then edit ACU_BACKEND_URL if needed
npm run dev                              # http://localhost:3000
```

Or, against the Docker Compose stack defined at the repo root:

```powershell
# In one terminal:
docker compose up
# Compose maps Django to host port 8001; point Next at it:
$env:ACU_BACKEND_URL = "http://127.0.0.1:8001"
npm run dev
```

## Scripts

| Script             | Purpose                                          |
| ------------------ | ------------------------------------------------ |
| `npm run dev`      | Local dev server with hot reload (port 3000).    |
| `npm run build`    | Production bundle.                               |
| `npm run start`    | Serve a previously-built bundle.                 |
| `npm run lint`     | Next.js ESLint preset.                           |
| `npm run typecheck`| `tsc --noEmit` against the strict tsconfig.      |

## Project layout

```
app/
├── layout.tsx              Root layout (font, metadata, body data attrs)
├── page.tsx                Composes the chat page (server component)
├── globals.css             1:1 copy of static/chatbot/css/styles.css + .sources-card
├── components/
│   ├── Chat.tsx            Top-level client component owning all state
│   ├── Sidebar.tsx         Conversation list + new chat button
│   ├── TopBar.tsx          Brand + tagline + "Yerel RAG" pill
│   ├── Welcome.tsx         Welcome card + suggestion buttons
│   ├── Composer.tsx        Textarea + send button (Enter to submit)
│   ├── MessageRow.tsx      User / assistant bubble rendering
│   ├── LoadingCard.tsx     Shimmer placeholder during /ask
│   └── SourcesCard.tsx     Adım 5.2: renders retrieved_chunks
├── lib/
│   ├── types.ts            Wire types mirroring docs/openapi.yaml
│   ├── api.ts              Browser-side fetchers (same-origin /api/*)
│   └── proxy.ts            Server-side fetch helper (cookie forwarding)
└── api/
    ├── ask/route.ts        POST → Django /api/v1/ask
    ├── conversations/route.ts        GET (list) / POST (create)
    └── conversations/[id]/route.ts   GET (detail)
public/
└── avatar.png              Same image legacy Django served from /static/.
```

## Why a server-side proxy

`app/api/ask/route.ts` and friends call Django on the server, not the browser.
That gives us:

- single-origin from the user's POV (no CORS preflight),
- Django session cookie travels through unchanged so the IDOR guarantee on
  conversation ownership keeps working end-to-end,
- `ACU_BACKEND_URL` never leaves the server.

The Django side (`/api/v1/ask`) is deliberately `@csrf_exempt` because the
intended caller is exactly this proxy — see `chatbot/api/v1/views.py` and
the architecture notes in `docs/openapi.yaml`.

## Updating the design

When you change anything visual:

1. Update both `app/globals.css` and `static/chatbot/css/styles.css` (the
   Django legacy stylesheet) so the two surfaces do not drift.
2. Mirror any HTML structure changes in `templates/index.html`.
3. Run `npm run typecheck` and `npm run build` before committing.

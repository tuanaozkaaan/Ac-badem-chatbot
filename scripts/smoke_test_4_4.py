"""Adım 4.4 — End-to-end smoke test + cosine threshold calibration.

Runs ten boundary-stretching questions through the live RAG stack:
parser → metadata-aware hybrid retrieval → Gemma 7B (Ollama) and emits a
single JSON file plus a printable summary table that the operator uses
to pick a defensible value for ``ACU_EMBEDDING_MIN_COSINE``.

Pure observability — does not mutate the DB.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import django  # noqa: E402

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "acu_chatbot.settings")
django.setup()

from chatbot.services.embedding import _retrieve_top_chunks_by_embedding
from chatbot.services.llm_client import ask_gemma
from chatbot.services.query_parser import parse_query

# Mix on purpose: course-code lookup, intent (officials/contact), semester filter,
# multi-intent, generic (no parser hits → global), and out-of-scope (Hemşirelik).
QUERIES: list[dict] = [
    {"id": "Q1", "label": "Spesifik ders kodu (CSE101 AKTS)",
     "q": "CSE101 dersinin AKTS kredisi kaçtır?"},
    {"id": "Q2", "label": "Intent: bölüm başkanı (officials)",
     "q": "Bilgisayar Mühendisliği bölüm başkanı kimdir?"},
    {"id": "Q3", "label": "Intent: iletişim (contact)",
     "q": "Bilgisayar Mühendisliği bölümünün iletişim bilgileri nelerdir?"},
    {"id": "Q4", "label": "Yarıyıl filtresi (semester=1)",
     "q": "Bilgisayar Mühendisliği 1. yarıyılında okutulan zorunlu dersler nelerdir?"},
    {"id": "Q5", "label": "Mezuniyet koşulları",
     "q": "Bilgisayar Mühendisliği bölümünden mezun olabilmek için ne yapmak gerekir?"},
    {"id": "Q6", "label": "Akademik personel listesi",
     "q": "Bilgisayar Mühendisliği bölümünde hangi akademik personel görev yapıyor?"},
    {"id": "Q7", "label": "Out-of-scope (Hemşirelik)",
     "q": "Hemşirelik bölümünün ders kataloğu nedir?"},
    {"id": "Q8", "label": "Genel arama / no parser hits",
     "q": "Acıbadem Üniversitesi nedir?"},
    {"id": "Q9", "label": "Multi-intent (kabul + mezuniyet)",
     "q": "Bilgisayar Mühendisliği için kabul koşulları ve mezuniyet koşulları nelerdir?"},
    {"id": "Q10", "label": "Kariyer / istihdam alanı",
     "q": "Bilgisayar Mühendisliği bölümünden mezun olanlar hangi alanlarda iş bulabilir?"},
]

K = 5


def build_prompt(question: str, hits: list[dict]) -> str:
    """Compact RAG prompt — keeps Gemma 7B latency in check."""
    blocks: list[str] = []
    for h in hits[:K]:
        meta = " | ".join(p for p in (h.get("title") or "", h.get("url") or "") if p)
        body = (h.get("text") or "").strip()
        if len(body) > 700:
            body = body[:700].rsplit("\n", 1)[0] + " ..."
        blocks.append(f"[{meta}]\n{body}" if meta else body)
    context = "\n\n---\n\n".join(blocks)
    return (
        "Aşağıdaki bağlamdan yararlanarak Türkçe yanıt ver. "
        "SADECE bağlamdaki bilgileri kullan, uydurma; bilgi yoksa "
        '"Bağlamda yeterli bilgi yok." de. 2-5 kısa cümle yaz.\n\n'
        f"Bağlam:\n{context}\n\n"
        f"Soru: {question}\nCevap:"
    )


def fmt_terms(terms: tuple[str, ...]) -> str:
    return ", ".join(terms) if terms else "<empty>"


def truncate(text: str, n: int = 110) -> str:
    text = (text or "").replace("\n", " / ").strip()
    return text if len(text) <= n else text[: n - 3] + "..."


def main() -> None:
    print(f"=== Adım 4.4 End-to-End Smoke Test ({len(QUERIES)} sorgu) ===", flush=True)
    print(f"K={K}, model=gemma:7b via Ollama", flush=True)

    results: list[dict] = []
    t_total = time.perf_counter()

    for case in QUERIES:
        qid = case["id"]
        q = case["q"]
        print("\n" + "-" * 78, flush=True)
        print(f"{qid}: {case['label']}", flush=True)
        print(f"Q: {q}", flush=True)

        # 1) Parser
        t0 = time.perf_counter()
        f = parse_query(q)
        parse_ms = (time.perf_counter() - t0) * 1000.0
        path = "hard" if not f.is_empty() else "global"
        print(
            f"  parse: {parse_ms:6.1f} ms  | path={path:6s} | "
            f"dept={f.department!r} fac={f.faculty!r} course={f.course_code!r} "
            f"sem={f.semester!r} ct={f.content_types} | terms={fmt_terms(f.matched_terms)}",
            flush=True,
        )

        # 2) Retrieval
        t0 = time.perf_counter()
        hits = _retrieve_top_chunks_by_embedding(q, k=K, filters=f)
        retrieve_ms = (time.perf_counter() - t0) * 1000.0
        scores = [float(h["score"]) for h in hits]
        score_max = max(scores) if scores else None
        score_min = min(scores) if scores else None
        print(
            f"  retrieve: {retrieve_ms:6.1f} ms | hits={len(hits)} | "
            f"max={score_max if score_max is None else round(score_max, 4)} "
            f"min={score_min if score_min is None else round(score_min, 4)}",
            flush=True,
        )
        for i, h in enumerate(hits, 1):
            print(
                f"    #{i} score={h['score']:.4f} chunk_id={h['chunk_id']:>4d} "
                f":: {truncate(h.get('title') or '', 50)} :: {truncate(h.get('text') or '', 90)}",
                flush=True,
            )

        # 3) LLM
        if hits:
            prompt = build_prompt(q, hits)
            t0 = time.perf_counter()
            answer = ask_gemma(prompt)
            llm_ms = (time.perf_counter() - t0) * 1000.0
        else:
            prompt = ""
            answer = "[NO HITS — LLM SKIPPED]"
            llm_ms = 0.0
        print(f"  llm:      {llm_ms:6.0f} ms | answer_chars={len(answer)}", flush=True)
        print(f"  ANSWER:\n{answer}", flush=True)

        results.append(
            {
                "id": qid,
                "label": case["label"],
                "question": q,
                "path": path,
                "parser": {
                    "department": f.department,
                    "faculty": f.faculty,
                    "course_code": f.course_code,
                    "semester": f.semester,
                    "content_types": list(f.content_types),
                    "matched_terms": list(f.matched_terms),
                },
                "timing_ms": {
                    "parse": round(parse_ms, 2),
                    "retrieve": round(retrieve_ms, 2),
                    "llm": round(llm_ms, 0),
                    "total": round(parse_ms + retrieve_ms + llm_ms, 0),
                },
                "scores": [round(s, 4) for s in scores],
                "score_max": None if score_max is None else round(score_max, 4),
                "score_min": None if score_min is None else round(score_min, 4),
                "hits_preview": [
                    {
                        "rank": i,
                        "chunk_id": h["chunk_id"],
                        "score": round(float(h["score"]), 4),
                        "title": h.get("title") or "",
                        "snippet": truncate(h.get("text") or "", 140),
                    }
                    for i, h in enumerate(hits, 1)
                ],
                "answer": answer,
            }
        )

    total_ms = (time.perf_counter() - t_total) * 1000.0

    out_path = Path("data/scraped/smoke_test_4_4.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out_path}  (toplam {total_ms / 1000.0:.1f}s)", flush=True)

    # Summary table
    print("\n" + "=" * 92, flush=True)
    print("=== Cosine & Latency Summary ===", flush=True)
    print(
        f"{'ID':4s} {'Path':6s} {'Parse':>8s} {'Retr':>8s} {'LLM':>8s} {'Hits':>5s}  "
        f"{'Max':>7s} {'Min':>7s}  Label",
        flush=True,
    )
    for r in results:
        smax = "-" if r["score_max"] is None else f"{r['score_max']:.4f}"
        smin = "-" if r["score_min"] is None else f"{r['score_min']:.4f}"
        print(
            f"{r['id']:4s} {r['path']:6s} "
            f"{r['timing_ms']['parse']:>6.1f}ms {r['timing_ms']['retrieve']:>6.1f}ms "
            f"{r['timing_ms']['llm']:>6.0f}ms {len(r['scores']):>5d}  "
            f"{smax:>7s} {smin:>7s}  {r['label']}",
            flush=True,
        )

    # Aggregate cosine stats for calibration recommendation.
    all_max = [r["score_max"] for r in results if r["score_max"] is not None]
    all_min = [r["score_min"] for r in results if r["score_min"] is not None]
    if all_max:
        print(
            f"\nCosine global → max-of-max={max(all_max):.4f} | min-of-max={min(all_max):.4f} "
            f"| min-of-min={min(all_min):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - smoke runner: surface trace, exit non-zero
        import traceback

        traceback.print_exc()
        sys.exit(1)

"""Context block helpers used by retrieval, the orchestrator, and extractive answers.

Responsibilities:
    * Splitting a retrieved context blob into blocks and labelling their source.
    * Scoring/selecting blocks for the final prompt under length constraints.
    * Light heuristics that gate "context relevant?" and "thread is about Acıbadem?".

Allowed dependency direction:
    context_select → services.intents, services.language
"""
from __future__ import annotations

import difflib
import re
from urllib.parse import urlparse

from chatbot.services.intents import (
    _detect_specific_faculty_focus,
    _extract_faculty_phrase,
)
from chatbot.services.language import (
    _TR_STOPWORDS,
    _ascii_fold_turkish,
    _extract_keywords,
)


def _split_context_blocks(context: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*---\s*\n", context or "") if p.strip()]
    return parts if parts else ([context.strip()] if (context or "").strip() else [])


def _block_matches_faculty(block: str, focus: str | None) -> bool:
    if not focus:
        return True
    b = _ascii_fold_turkish(block or "")
    if focus == "saglik_bilimleri":
        return "saglik bilimleri fakultesi" in b or "saglik bilimleri" in b
    if focus == "muhendislik_doga":
        return "muhendislik ve doga bilimleri" in b
    return True


def _extract_block_source_label(block: str) -> str:
    first = (block.splitlines() or [""])[0].strip()
    if first.startswith("[") and first.endswith("]"):
        meta = first[1:-1]
        parts = [p.strip() for p in meta.split("|") if p.strip()]
        for p in parts:
            if p.startswith("http://") or p.startswith("https://"):
                up = urlparse(p)
                tail = up.path.strip("/").split("/")[-1] or up.netloc
                return tail[:80]
        if parts:
            return parts[0][:80]
    return "local_text"


def _strip_urls_plain_text(answer: str) -> str:
    """Remove hyperlinks so answers stay plain text (markdown + bare URLs)."""
    s = (answer or "").strip()
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return re.sub(r" {2,}", " ", s).strip()


def _answer_is_stock_no_info(answer: str) -> bool:
    al = re.sub(r"\s+", " ", (answer or "").strip().lower())
    return al.startswith("bu konuda elimde net bir bilgi bulunamad") or al.startswith(
        "i couldn't find clear information about this"
    )


def _looks_acibadem_related(question: str) -> bool:
    q = _ascii_fold_turkish(question or "")
    return any(
        k in q
        for k in (
            "acibadem",
            "acibadem universitesi",
            "acibadem university",
        )
    )


def _thread_user_text_blob(conv, limit: int = 14) -> str:
    # Lazy import to keep this module free of Django ORM at import-time.
    from chatbot.models import Message

    qs = Message.objects.filter(conversation=conv, role=Message.ROLE_USER).order_by("-id")[:limit]
    return " ".join((m.content or "") for m in qs)


def _thread_suggests_acibadem_topic(conv) -> bool:
    return _looks_acibadem_related(_thread_user_text_blob(conv))


def _context_likely_relevant(question: str, context: str) -> bool:
    """
    Heuristic: question keywords (ASCII-folded) appear in folded context — suggests
    the model should answer from context instead of a generic refusal.
    """
    if not (context or "").strip():
        return False
    ctxf = _ascii_fold_turkish(context)
    qf = _ascii_fold_turkish(question or "")
    overlap_skip = _TR_STOPWORDS | frozenset(
        "nedir kimdir nerede nasil nasıl hangi niye niçin nicin bana bizi boyle".split()
    )
    terms = [w for w in re.findall(r"[\w]+", qf) if len(w) >= 4 and w not in overlap_skip][:16]
    long_hits = sum(1 for t in terms if len(t) >= 8 and t in ctxf)
    short_hits = sum(1 for t in terms if 4 <= len(t) < 8 and t in ctxf)
    if long_hits >= 1 or short_hits >= 1:
        return True

    # Cross-language guard: TR question may map to EN context (or vice versa).
    # Keep this broad but safe so we do not drop useful context too early.
    if _looks_acibadem_related(question) and (
        "acibadem" in ctxf or "acıbadem" in (context or "").lower()
    ):
        time_intent = any(
            k in qf
            for k in ("ne zaman", "when", "kuruldu", "established", "basladi", "started")
        )
        if time_intent and (re.search(r"\b(19|20)\d{2}\b", context or "") is not None):
            return True
        location_intent = any(k in qf for k in ("adres", "address", "kampus", "campus", "nerede", "where"))
        if location_intent and any(k in ctxf for k in ("atasehir", "ataşehir", "istanbul", "kayisdagi", "kayışdağı", "cad", "no:")):
            return True
        # Generic fallback for Acibadem-related questions when context clearly contains Acibadem text.
        return True

    return False


def _select_context_for_llm(
    question: str,
    context: str,
    *,
    max_chunks: int = 5,
    max_chars: int = 5000,
) -> tuple[str, list[str], int]:
    blocks = _split_context_blocks(context)
    retrieved_count = len(blocks)
    if not blocks:
        return "", [], 0

    keywords = _extract_keywords(question)[:14]
    q_fold = _ascii_fold_turkish(question or "")
    faculty_focus = _detect_specific_faculty_focus(question)
    faculty_phrase = _extract_faculty_phrase(question)

    # Remove exact/near-duplicate chunks.
    deduped: list[str] = []
    fingerprints: list[str] = []
    for block in blocks:
        fp = re.sub(r"\s+", " ", _ascii_fold_turkish(block)).strip()
        if not fp:
            continue
        duplicate = False
        for seen in fingerprints:
            if fp == seen:
                duplicate = True
                break
            if difflib.SequenceMatcher(a=fp[:800], b=seen[:800]).ratio() >= 0.92:
                duplicate = True
                break
        if duplicate:
            continue
        deduped.append(block)
        fingerprints.append(fp)

    if faculty_focus:
        focused = [b for b in deduped if _block_matches_faculty(b, faculty_focus)]
        if focused:
            deduped = focused
    if faculty_phrase:
        strict_focused = [b for b in deduped if faculty_phrase in _ascii_fold_turkish(b)]
        if strict_focused:
            deduped = strict_focused

    scored: list[tuple[float, str]] = []
    for block in deduped:
        bf = _ascii_fold_turkish(block)
        hit_score = 0.0
        for kw in keywords:
            kf = _ascii_fold_turkish(kw)
            if kf and kf in bf:
                hit_score += 1.0
        if faculty_focus and _block_matches_faculty(block, faculty_focus):
            hit_score += 3.5
        if any(x in bf for x in ("duyuru", "etkinlik")) and not any(k in q_fold for k in ("duyuru", "etkinlik")):
            hit_score -= 0.8
        scored.append((hit_score, block))
    scored.sort(key=lambda x: x[0], reverse=True)

    selected: list[str] = []
    selected_sources: list[str] = []
    total_chars = 0
    for score, block in scored:
        if score < 0 and selected:
            continue
        chunk_len = len(block)
        if selected and total_chars + chunk_len > max_chars:
            continue
        selected.append(block)
        selected_sources.append(_extract_block_source_label(block))
        total_chars += chunk_len
        if len(selected) >= max_chunks or total_chars >= max_chars:
            break

    selected_context = "\n\n---\n\n".join(selected).strip()
    if not selected_context:
        return "", [], retrieved_count
    if len(selected_context) > max_chars:
        selected_context = selected_context[:max_chars].rsplit("\n", 1)[0].strip()
    if not _context_likely_relevant(question, selected_context):
        return "", selected_sources, retrieved_count
    return selected_context, selected_sources, retrieved_count


def _faculty_richness(tf: str) -> int:
    """
    Tam fakülte listesi sayfalarını ayırt etmek için skor (yüksek = daha çok farklı birim adı geçiyor).
    Yalnızca '...Fakültesi' tekrarı tek başına yüksek skor vermez; farklı ipuçları (tıp, mühendislik vb.) ağırlıklıdır.
    """
    if not tf:
        return 0
    hints = (
        "tip fakulte",
        "eczacilik fakulte",
        "muhendislik ve doga",
        "muhendislik fakulte",
        "saglik bilimleri fakulte",
        "guzel sanatlar",
        "hukuk fakulte",
        "iletisim fakulte",
        "dis hekimligi",
        "beslenme ve diyetetik",
        "fizyoterapi",
        "hemsirelik",
        "egitim bilimleri",
        "yabanci diller",
        "yuksekokul",
        "onlisans",
        "meslek yuksekokul",
    )
    hint_hits = sum(1 for h in hints if h in tf)
    # Tekrarlayan "fakultesi" kelimesi en fazla +5 katkı; asıl ağırlık ipuçlarında.
    return int(hint_hits * 14 + min(5, tf.count("fakultesi")))


__all__ = [
    "_split_context_blocks",
    "_block_matches_faculty",
    "_extract_block_source_label",
    "_strip_urls_plain_text",
    "_answer_is_stock_no_info",
    "_looks_acibadem_related",
    "_thread_user_text_blob",
    "_thread_suggests_acibadem_topic",
    "_context_likely_relevant",
    "_select_context_for_llm",
    "_faculty_richness",
]

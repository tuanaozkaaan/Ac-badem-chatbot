import json
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import requests
from django.db.models import Q
from django.http import JsonResponse
from django.utils import timezone
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

# Crawled chunks often omit the postal block; official page:
# https://acibadem.edu.tr/kayit/iletisim/ulasim
OFFICIAL_CAMPUS_ADDRESS_BLOCK = (
    "[Resmî kampüs adresi ve iletişim — Acıbadem Üniversitesi; "
    "kaynak: acibadem.edu.tr/kayit/iletisim/ulasim]\n"
    "Kerem Aydınlar Kampüsü, Kayışdağı Cad. No:32, 34752 Ataşehir/İstanbul\n"
    "Telefon: 0216 500 44 44, 0216 576 50 76\n"
    "E-posta: info@acibadem.edu.tr\n"
)


def _ascii_fold_turkish(s: str) -> str:
    """Lowercase + map Turkish letters so 'acıbadem' and 'acibadem' both match 'acibadem'."""
    q = (s or "").lower()
    return (
        q.replace("ı", "i")
        .replace("ğ", "g")
        .replace("ü", "u")
        .replace("ş", "s")
        .replace("ö", "o")
        .replace("ç", "c")
        .replace("\u0307", "")
    )


def _wants_postal_address_detail(q_lower: str) -> bool:
    if any(x in q_lower for x in ("address", "postal", "zip code", "zipcode")):
        return True
    if "adres" in q_lower:
        return True
    return any(
        p in q_lower
        for p in (
            "tam adres",
            "posta kodu",
            "cadde",
            "sokak",
            "detayli adres",
            "detaylı adres",
            "tam konum",
            "full address",
        )
    )


def _canonical_campus_address_reply(is_tr: bool) -> str:
    if is_tr:
        return (
            "Acıbadem Üniversitesi’nin posta tarzı kampüs adresi şöyledir:\n\n"
            "Kerem Aydınlar Kampüsü, Kayışdağı Cad. No:32, 34752 Ataşehir/İstanbul\n\n"
            "Telefon: 0216 500 44 44, 0216 576 50 76\n"
            "E-posta: info@acibadem.edu.tr"
        )
    return (
        "The official postal-style campus address is:\n\n"
        "Kerem Aydınlar Campus, Kayışdağı Avenue No:32, 34752 Ataşehir, Istanbul, Turkiye\n\n"
        "Phone: +90 216 500 44 44, +90 216 576 50 76\n"
        "Email: info@acibadem.edu.tr"
    )


def _strip_urls_plain_text(answer: str) -> str:
    """Remove hyperlinks so answers stay plain text (markdown + bare URLs)."""
    s = (answer or "").strip()
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    s = re.sub(r"https?://\S+", "", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return re.sub(r" {2,}", " ", s).strip()


_TR_STOPWORDS = {
    "ve",
    "veya",
    "ile",
    "için",
    "icin",
    "nedir",
    "ne",
    "nerede",
    "nasıl",
    "nasil",
    "hangi",
    "kim",
    "kaç",
    "kac",
    "mı",
    "mi",
    "mu",
    "mü",
    "bu",
    "şu",
    "su",
    "o",
    "bir",
    "da",
    "de",
    "ama",
    "fakat",
    "gibi",
    "üniversite",
    "universite",
    "üniversitesi",
    "universitesi",
    "acibadem",
    "acıbadem",
}


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
    from chatbot.models import Message

    qs = Message.objects.filter(conversation=conv, role=Message.ROLE_USER).order_by("-id")[:limit]
    return " ".join((m.content or "") for m in qs)


def _thread_suggests_acibadem_topic(conv) -> bool:
    return _looks_acibadem_related(_thread_user_text_blob(conv))


def _detect_language(question: str) -> str:
    """
    Heuristic language detection for TR vs EN.
    Returns: "tr" or "en"
    """
    q = (question or "").strip().lower()
    if not q:
        return "tr"

    # Turkish-specific characters are a strong signal.
    if any(ch in q for ch in ("ç", "ğ", "ı", "ö", "ş", "ü")):
        return "tr"

    tr_markers = {
        "mı",
        "mi",
        "mu",
        "mü",
        "nedir",
        "nerede",
        "nasıl",
        "nasil",
        "kimdir",
        "hangisi",
        "anlat",
        "fakülte",
        "fakulte",
        "bölüm",
        "bolum",
        "üniversite",
        "universite",
        "adres",
        "ulaşım",
        "ulasim",
    }
    en_markers = {
        "what",
        "where",
        "how",
        "who",
        "tell",
        "explain",
        "faculty",
        "department",
        "university",
        "address",
        "campus",
        "admission",
        "apply",
    }

    tokens = set(q.replace("?", " ").replace(".", " ").replace(",", " ").split())
    if tokens & tr_markers:
        return "tr"
    if tokens & en_markers:
        return "en"

    # Default: Turkish (project is TR-first).
    return "tr"


def _looks_turkish(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    if any(ch in t for ch in ("ç", "ğ", "ı", "ö", "ş", "ü")):
        return True
    # Common Turkish function words
    tr_words = (" ve ", " ile ", " için ", " adres", " üniversite", " kampüs", " istanbul", " türkiye", " nedir", " nerede")
    hits = sum(1 for w in tr_words if w in f" {t} ")
    return hits >= 2


def _looks_english(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    en_words = (" the ", " and ", " address", " university", " campus", " contact", " located", " is ")
    hits = sum(1 for w in en_words if w in f" {t} ")
    return hits >= 2


def _translate_answer(answer: str, target_lang: str) -> str:
    """
    Translate an already-produced answer to the target language ("tr" or "en").
    Must preserve meaning and avoid adding new facts.
    """
    if target_lang not in ("tr", "en"):
        return answer
    if not (answer or "").strip():
        return answer
    to_label = "Turkish" if target_lang == "tr" else "English"
    prompt = f"""Translate the text below to {to_label}.
Rules:
- Preserve meaning exactly.
- Do not add any new information.
- Output only the translation (no quotes, no extra text).

Text:
{answer}
"""
    translated = ask_gemma(prompt)
    return (translated or "").strip() or answer

def _extract_keywords(question: str) -> list[str]:
    tokens = [
        t.strip(".,!?;:()[]{}\"'").lower()
        for t in (question or "").split()
        if len(t.strip(".,!?;:()[]{}\"'")) >= 3
    ]
    keywords = [t for t in tokens if t not in _TR_STOPWORDS]
    # keep order, de-dup
    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


def retrieve_context(question: str, k: int = 5) -> str:
    """
    Scoring-based retrieval over PageChunk using ONLY Python logic (no embeddings).
    Weights:
      - title: highest
      - section: medium
      - chunk_text: lowest
    Also applies intent-based boosts/penalties for admissions vs internship queries.

    Candidate chunks are scored (including negative/zero), then top *k* by score are returned.
    To avoid scanning the entire table on large ingests, candidates are narrowed with
    `icontains` OR filters on keywords (plus a short prefix for long tokens) and capped by
    `RETRIEVE_MAX_CANDIDATES` (default 6000). If no keyword matches, the most recently updated
    chunks up to that cap are used instead.
    """
    keywords = _extract_keywords(question)
    # Question words widen OR-matches without helping entity questions ("kimdir" on half the site).
    lookup_noise = frozenset(
        {"kimdir", "nedir", "midir", "nerede", "nasıl", "nasil", "hangi", "neden", "niçin", "nicin", "kim", "ne"}
    )
    keywords = [k for k in keywords if k not in lookup_noise]
    q_lower = (question or "").lower()

    # Multi-word phrases (not produced by whitespace tokenization) improve DB icontains narrowing.
    extra_lookup_terms: list[str] = []
    if "bilgisayar mühendisliği" in q_lower or "bilgisayar muhendisligi" in q_lower:
        extra_lookup_terms.extend(
            ["bilgisayar mühendisliği", "bilgisayar muhendisligi", "bilgisayar mühendis", "bilgisayar muhendis"]
        )
    if "bölüm başkanı" in q_lower or "bolum baskani" in q_lower:
        extra_lookup_terms.extend(["bölüm başkanı", "bolum baskani", "bölüm başkan", "bolum baskan"])

    # Intent detection (priority matters):
    # - If question mentions staj/internship => internship intent
    # - Else if question mentions apply/admission terms => general admission intent
    internship_intent = any(t in q_lower for t in ("staj", "internship"))
    admission_intent = (not internship_intent) and any(
        t in q_lower for t in ("başvuru", "basvuru", "admission", "apply", "application")
    )
    address_intent = any(
        t in q_lower
        for t in (
            "adres",
            "address",
            "kampüs",
            "kampus",
            "campus",
            "konum",
            "location",
            "ulaşım",
            "ulasim",
            "nerede",
            "nasıl gid",
            "nasil gid",
        )
    )
    faculty_head_intent = any(
        t in q_lower
        for t in (
            "bölüm başkanı",
            "bolum baskani",
            "bölüm başkan",
            "bolum baskan",
            "program başkanı",
            "program baskani",
            "department head",
            "department chair",
            "müdür",
            "mudur",
        )
    )
    if address_intent:
        # Keep these specific (avoid broad "İstanbul" OR matches that flood candidates).
        extra_lookup_terms.extend(
            [
                "kerem aydınlar",
                "kerem aydinlar",
                "iletişim ve ulaşım",
                "iletisim ve ulasim",
                "atköy",
                "atkoy",
                "inönü",
                "inonu",
                "ulaşım",
                "ulasim",
            ]
        )

    admission_boost_terms = {
        "admission",
        "başvuru",
        "basvuru",
        "application",
        "apply",
        "kabul",
        "requirements",
        "öğrenci kabul",
        "ogrenci kabul",
    }
    admission_penalize_terms = {
        "staj",
        "internship",
        "staj başvurusu",
        "staj basvurusu",
        "career",
        "kariyer",
    }

    internship_boost_terms = {
        "staj",
        "internship",
        "staj başvurusu",
        "staj basvurusu",
        "staj yönergesi",
        "staj yonergesi",
        "staj defteri",
        "staj komisyonu",
    }
    internship_penalize_terms = {
        "admission",
        "başvuru",
        "basvuru",
        "application",
        "apply",
        "öğrenci kabul",
        "ogrenci kabul",
        "requirements",
        "kabul",
    }

    noisy_penalize_terms = {"duyuru", "haber", "etkinlik", "announcement", "news", "event"}
    address_boost_terms = {
        "adres",
        "address",
        "iletişim",
        "iletisim",
        "ulaşım",
        "ulasim",
        "kampüs",
        "kampus",
        "kampüsü",
        "kampusu",
        "campus",
        "yerleşke",
        "yerleske",
    }
    address_penalize_terms = {
        # Common in program pages and tends to match "adresine" (email address) rather than campus address.
        "referans",
        "transkript",
        "ales",
        "başvuru",
        "basvuru",
        "application",
    }

    def _count_hits(hay: str, needle: str) -> int:
        if not hay or not needle:
            return 0
        return hay.count(needle)

    def _score_row(row) -> int:
        title = (row.title or "").lower()
        section = (row.section or "").lower()
        text = (row.chunk_text or "").lower()
        url = (getattr(row, "url", None) or "").lower()

        score = 0

        # Keyword scoring: title highest, section medium, chunk_text lowest.
        for kw in keywords[:10]:
            score += 8 * _count_hits(title, kw)
            score += 4 * _count_hits(section, kw)
            score += 2 * _count_hits(text, kw)

        if admission_intent:
            for t in admission_boost_terms:
                if t in title:
                    score += 40
                if t in section:
                    score += 24
                if t in text:
                    score += 14

            for t in admission_penalize_terms:
                if t in title:
                    score -= 90
                if t in section:
                    score -= 60
                if t in text:
                    score -= 40

        elif internship_intent:
            for t in internship_boost_terms:
                if t in title:
                    score += 40
                if t in section:
                    score += 24
                if t in text:
                    score += 14

            for t in internship_penalize_terms:
                if t in title:
                    score -= 80
                if t in section:
                    score -= 55
                if t in text:
                    score -= 35

        for t in noisy_penalize_terms:
            if t in title:
                score -= 18
            if t in section:
                score -= 12
            if t in text:
                score -= 8

        if address_intent:
            for t in address_boost_terms:
                if t in title:
                    score += 36
                if t in section:
                    score += 22
                if t in text:
                    score += 10

            for t in address_penalize_terms:
                if t in title:
                    score -= 40
                if t in section:
                    score -= 24
                if t in text:
                    score -= 14

            # Heuristic: lots of emails often means we matched "adresine" (email address).
            email_hits = text.count("@")
            if email_hits >= 2 and not any(x in text for x in ("kampüs", "kampus", "campus", "ataşehir", "atasehir")):
                score -= 60

            # Strongly prefer actual contact/transport pages when asking for address/location.
            if any(p in url for p in ("/iletisim", "/contact", "/ulasim", "/adres", "/yerleske", "/kampus", "/kampus/")):
                score += 140
            if ("iletişim" in title) or ("iletisim" in title) or ("contact" in title):
                score += 140

            # Academic program pages are usually not about physical location.
            if "/akademik/" in url and not any(p in url for p in ("/iletisim", "/contact", "/ulasim", "/adres")):
                score -= 35

            geo_markers = (
                "ataşehir",
                "atasehir",
                "istanbul",
                "i̇stanbul",
                "caddesi",
                "cad.",
                "mahalle",
                "mah.",
                "posta kodu",
                "pk.",
                "inönü",
                "inonu",
                "atköy",
                "atkoy",
            )
            geo_hits = sum(1 for g in geo_markers if g in text)
            if geo_hits >= 2:
                score += 110
            elif geo_hits == 1:
                score += 50
            if "no:" in text or " no:" in text:
                score += 28
            # Down-rank snippets that only place an office on a floor without street/district clues.
            if "kat" in text and any(o in text for o in ("ofis", "office", "acumed", "büro", "biro")):
                if not any(s in text for s in ("ataşehir", "atasehir", "istanbul", "caddesi", "mahalle", "inönü", "inonu")):
                    score -= 95

        if faculty_head_intent:
            if any(x in text for x in ("başkan", "baskan", "müdür", "mudur", "chair", "head of")):
                score += 28
            if any(x in title for x in ("başkan", "baskan", "müdür", "mudur", "chair")):
                score += 55
            if any(x in section for x in ("başkan", "baskan", "müdür", "mudur")):
                score += 35
            if "/akademik/" in url:
                score += 18

        # When user names a specific engineering/CS department, avoid ranking unrelated "bölüm başkanı" pages.
        cs_dept_focus = faculty_head_intent and any(
            t in q_lower for t in ("bilgisayar", "yazılım", "yazilim", "computer", "mühendislik", "muhendislik")
        )
        if cs_dept_focus:
            if any(t in text for t in ("bilgisayar", "yazılım", "yazilim", "computer science", "computer")):
                score += 55
            if any(t in title for t in ("bilgisayar", "yazılım", "yazilim", "computer")):
                score += 70
            if any(t in url for t in ("bilgisayar", "yazilim", "computer", "mühendislik", "muhendislik")):
                score += 45
            if "bilgisayar-programciligi" in url or "bilgisayar-programcılığı" in url:
                score += 85
            if ("sağlık yönetimi" in text or "saglik yonetimi" in text) and "bilgisayar" not in text and "yazılım" not in text:
                score -= 95
            if ("onlisans" in url or "onlisans" in text) and "bilgisayar" not in text:
                score -= 35
            for bad in (
                "saglik-yonetimi",
                "saglik-hizmetleri",
                "radyoterapi",
                "yabanci-diller",
                "ingilizce-hazirlik",
            ):
                if bad in url and "bilgisayar-programciligi" not in url and "muhendislik" not in url:
                    score -= 140

        return score

    # 1) Prefer DB chunks if available — score a bounded candidate set, then take top k by score.
    try:
        from chatbot.models import PageChunk

        if PageChunk.objects.exists():
            max_candidates = int(os.environ.get("RETRIEVE_MAX_CANDIDATES", "6000"))
            base_qs = PageChunk.objects.only(
                "chunk_text", "title", "section", "url", "source_type", "updated_at"
            ).order_by("-updated_at")

            lookup_terms: list[str] = []
            seen_terms: set[str] = set()
            for kw in keywords[:12]:
                if len(kw) < 3:
                    continue
                if kw not in seen_terms:
                    seen_terms.add(kw)
                    lookup_terms.append(kw)
                # Longer root for DB icontains (avoid e.g. "başkanı"[:5] -> "başka" false matches).
                if len(kw) >= 7:
                    plen = min(12, max(6, len(kw) - 1))
                    root = kw[:plen]
                    if len(root) >= 5 and root != kw and root not in seen_terms:
                        seen_terms.add(root)
                        lookup_terms.append(root)
                if len(lookup_terms) >= 22:
                    break

            for t in extra_lookup_terms:
                tt = (t or "").strip()
                if len(tt) >= 3 and tt not in seen_terms:
                    seen_terms.add(tt)
                    lookup_terms.append(tt)

            q_filter = Q()
            if lookup_terms:
                for term in lookup_terms:
                    q_filter |= (
                        Q(chunk_text__icontains=term)
                        | Q(title__icontains=term)
                        | Q(section__icontains=term)
                        | Q(url__icontains=term)
                    )

            candidate_qs = base_qs
            if address_intent:
                # Prefer real contact / transport / district lines instead of random "İstanbul" mentions.
                addr_anchor = (
                    Q(url__icontains="iletisim")
                    | Q(url__icontains="ulasim")
                    | Q(url__icontains="contact")
                    | Q(url__icontains="/adres")
                    | Q(chunk_text__icontains="ataşehir")
                    | Q(chunk_text__icontains="atasehir")
                    | Q(chunk_text__icontains="inönü")
                    | Q(chunk_text__icontains="inonu")
                    | Q(chunk_text__icontains="kerem aydınlar")
                    | Q(chunk_text__icontains="kerem aydinlar")
                )
                addr_qs = base_qs.filter(addr_anchor)
                if addr_qs.exists():
                    if lookup_terms:
                        merged = addr_qs.filter(q_filter)
                        candidate_qs = merged if merged.exists() else addr_qs
                    else:
                        candidate_qs = addr_qs
                elif lookup_terms:
                    narrowed = base_qs.filter(q_filter)
                    candidate_qs = narrowed if narrowed.exists() else base_qs[:max_candidates]
                else:
                    candidate_qs = base_qs[:max_candidates]
            elif lookup_terms:
                narrowed = base_qs.filter(q_filter)
                candidate_qs = narrowed if narrowed.exists() else base_qs[:max_candidates]
            else:
                candidate_qs = base_qs[:max_candidates]

            candidate_qs = candidate_qs[:max_candidates]

            scored_all: list[tuple[int, object, object]] = []
            for row in candidate_qs.iterator(chunk_size=500):
                s = _score_row(row)
                updated = row.updated_at or ""
                scored_all.append((s, updated, row))

            scored_all.sort(key=lambda x: (x[0], x[1]), reverse=True)
            top_score = scored_all[0][0] if scored_all else 0
            # Names / short queries often get low or negative scores after penalties; still return
            # best-effort chunks so "kimdir?" style questions can use weak DB hits.
            weak_hits_ok = any(len((kw or "").strip()) >= 3 for kw in keywords[:8])
            if top_score <= 0 and not weak_hits_ok:
                return ""

            logger.info("retrieve_context(db): question=%r candidates=%s", question, len(scored_all))
            for rank, (s, _u, row) in enumerate(scored_all[:k], start=1):
                logger.info("retrieve_context(db): #%s score=%s title=%r", rank, s, row.title)

            top_rows = [row for _, _, row in scored_all[:k]]
            blocks: list[str] = []
            for c in top_rows:
                meta = " | ".join(
                    [p for p in [c.title or "", c.section or "", c.source_type or "", c.url or ""] if p]
                )
                blocks.append(f"[{meta}]\n{c.chunk_text}" if meta else c.chunk_text)

            return "\n\n---\n\n".join(blocks).strip()
    except Exception:
        pass

    # 2) Fallback: local /data/*.txt — include all scored lines, take top k even if low score.
    try:
        from rag.document_loader import load_text_documents
        from rag.text_splitter import split_into_chunks
    except Exception:
        return ""

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = str(base_dir / "data")
    try:
        docs = load_text_documents(data_dir)
        chunks = split_into_chunks(docs, chunk_size=900, chunk_overlap=150)
    except Exception:
        return ""

    scored_txt: list[tuple[int, int, str]] = []
    for idx, ch in enumerate(chunks):
        text_l = ch.lower()
        score = 0
        for kw in keywords[:10]:
            score += text_l.count(kw)
        if admission_intent:
            for t in admission_boost_terms:
                if t in text_l:
                    score += 4
            for t in admission_penalize_terms:
                if t in text_l:
                    score -= 10
        elif internship_intent:
            for t in internship_boost_terms:
                if t in text_l:
                    score += 4
            for t in internship_penalize_terms:
                if t in text_l:
                    score -= 10

        for t in noisy_penalize_terms:
            if t in text_l:
                score -= 2
        scored_txt.append((score, -idx, ch))

    scored_txt.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = [c for _, _, c in scored_txt[:k]]
    return "\n\n---\n\n".join(top).strip()



def ask_gemma(prompt: str) -> str:
    try:
        base_url = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        model = (os.environ.get("OLLAMA_MODEL") or "gemma2:2b").strip()
        # Keep responses fast/compact by default; can be overridden via env.
        num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))
        temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
        # Keep model loaded in VRAM/RAM between requests (e.g. "10m", "0" to unload). Empty = omit.
        keep_alive = (os.environ.get("OLLAMA_KEEP_ALIVE") or "").strip()
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
            },
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=300,
        )
        if response.status_code == 404:
            body = (response.text or "").strip()[:400]
            return (
                "Gemma error: Ollama returned 404 (usually the model is not downloaded yet). "
                f"Run once: docker compose exec ollama ollama pull {model} "
                f"(or locally: ollama pull {model}). "
                f"Ollama said: {body or 'empty body'}"
            )
        response.raise_for_status()
        data = response.json()
        return (data["response"] or "").strip()
    except KeyError:
        return "Gemma error: Missing 'response' field in Ollama JSON."
    except Exception as e:
        return f"Gemma error: {str(e)}"


@lru_cache(maxsize=1)
def _bootstrap_rag() -> None:
    """RAG sistemini (yapay zekayı) bir kez yükler."""
    from backend.api import init_rag
    init_rag(model_path=os.environ.get("MODEL_PATH") or None)

def health(_request):
    return JsonResponse({"status": "ok"})


def _conversation_title_from_question(question: str, max_len: int = 80) -> str:
    t = " ".join((question or "").split())
    if not t:
        return "Yeni sohbet"
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "…"


def _touch_conversation_updated_at(conv) -> None:
    from chatbot.models import Conversation

    Conversation.objects.filter(pk=conv.pk).update(updated_at=timezone.now())


def _resolve_conversation(body: dict):
    from chatbot.models import Conversation

    raw = body.get("conversation_id")
    if raw in (None, "", False):
        return Conversation.objects.create(title=""), None
    try:
        cid = int(raw)
    except (TypeError, ValueError):
        return None, JsonResponse({"detail": "Invalid conversation_id."}, status=400)
    conv = Conversation.objects.filter(pk=cid).first()
    if not conv:
        return None, JsonResponse({"detail": "Conversation not found."}, status=404)
    return conv, None


def _persist_assistant_reply(conv, text: str, *, status: int = 200, as_detail: bool = False) -> JsonResponse:
    from chatbot.models import Message

    Message.objects.create(conversation=conv, role=Message.ROLE_ASSISTANT, content=text)
    _touch_conversation_updated_at(conv)
    payload: dict = {"conversation_id": conv.pk}
    if as_detail:
        payload["detail"] = text
    else:
        payload["answer"] = text
    return JsonResponse(payload, status=status)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def ask(request):
    # --- 1. ADIM: Arkadaşının Tasarımını Göster (Tarayıcıdan girince) ---
    if request.method == "GET":
        return render(request, "index.html") 

    # --- 2. ADIM: Gelen Mesajı Gemma'ya Gönder (Chat kutusuna yazınca) ---
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON."}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"detail": "Question cannot be empty."}, status=400)
    question = unicodedata.normalize("NFC", question)

    conv, conv_err = _resolve_conversation(body)
    if conv_err is not None:
        return conv_err

    from chatbot.models import Message

    Message.objects.create(conversation=conv, role=Message.ROLE_USER, content=question)
    if not (conv.title or "").strip():
        conv.title = _conversation_title_from_question(question)
        conv.save(update_fields=["title"])

    try:
        lang = _detect_language(question)
        is_tr = lang == "tr"
        no_info_msg = (
            "Bu konuda elimde net bir bilgi bulunamadı."
            if is_tr
            else "I couldn't find clear information about this."
        )

        # ASCII-fold so Turkish chars and .lower() quirks cannot skip the postal shortcut.
        q_fold = _ascii_fold_turkish(question)
        address_intent = any(
            t in q_fold
            for t in (
                "adres",
                "address",
                "kampus",
                "campus",
                "konum",
                "location",
                "ulasim",
                "nerede",
            )
        )
        if _wants_postal_address_detail(q_fold) and (
            "acibadem" in q_fold or _thread_suggests_acibadem_topic(conv)
        ):
            return _persist_assistant_reply(conv, _canonical_campus_address_reply(is_tr))

        context = retrieve_context(question, k=8 if address_intent else 5)
        if address_intent:
            ctx_body = (context or "").strip()
            context = (
                f"{OFFICIAL_CAMPUS_ADDRESS_BLOCK}\n{ctx_body}".strip()
                if ctx_body
                else OFFICIAL_CAMPUS_ADDRESS_BLOCK
            )
            # Small models latch onto random URLs from retrieved chunks; keep plain text for the model.
            context = re.sub(r"https?://\S+", "", context)
            context = re.sub(r" {2,}", " ", context)
            context = re.sub(r"\n{3,}", "\n\n", context).strip()
        if not context:
            return _persist_assistant_reply(
                conv,
                (
                    (
                        "Bu soru için veritabanımda eşleşen metin bulamadım; doğrulanmamış bilgi uydurmam. "
                        "Sorunuzu Acıbadem Üniversitesi ile ilgili anahtar kelimelerle (bölüm, program, kişi adı, konu) "
                        "yeniden sorabilir veya resmî web sitesinden doğrulayabilirsiniz."
                    )
                    if is_tr
                    else (
                        "I could not find matching text in my database for this question, so I cannot invent facts. "
                        "Try rephrasing with Acıbadem University–related keywords (department, program, person, topic) "
                        "or verify on the official website."
                    )
                ),
            )

        # Truncate context to keep latency manageable on small local models.
        max_context_chars = int(os.environ.get("DJANGO_MAX_CONTEXT_CHARS", "4500"))
        if len(context) > max_context_chars:
            context = context[:max_context_chars].rsplit("\n", 1)[0].strip()

        answer_language_instruction = "Türkçe" if is_tr else "English"

        address_rules = ""
        if address_intent:
            address_rules = """
ADDRESS / LOCATION QUESTIONS:
- The context begins with an official postal-style campus block from the university's Contact/Transportation page. When the user asks for the school/campus address or location, state that full address clearly in your first sentence or first short paragraph (street, number, postal code, district, city).
- Do NOT answer with hyperlinks, markdown links, or bare URLs. Do not tell the user to "go to this link". Use plain text only.
- If additional context below describes metro/bus routes or a building/floor, add that after the postal address; do not replace the postal address with an indoor office line alone.
- Prefer the official postal-style campus address when it appears in the context (district, city, street/avenue, building number, postal code if any).
- If the context mixes a general campus address with an indoor office location (e.g. a unit on a specific floor), lead with the postal/campus address; mention the office only as secondary detail from the same context.
- Never treat an indoor office line as the full university address if a broader postal/campus line exists in the context.
"""

        prompt = f"""
You are a helpful university assistant.

LANGUAGE RULES:
- Detect the language of the user's question.
- Always answer in the SAME language as the user’s question.
- If the question is in English → answer in English.
- If the question is in Turkish → answer in Turkish.

RETRIEVAL RULES:
- The context may be in Turkish or English.
- You MUST use the context even if it is in a different language than the question.
- If needed, translate the relevant information before answering.

SCOPE / PROGRAM NAME:
- The user may say "bilgisayar mühendisliği" while the retrieved pages describe a closely related program (e.g. Bilgisayar Programcılığı önlisans). If the context contains a named program head / coordinator, answer with that person and clearly state which program/level the source refers to.
- Never invent a person or title that is not supported by the context.
{address_rules}
ANSWERING RULES:
- Do NOT hallucinate.
- If the answer exists in the context, use it clearly.
- If the context is in another language, translate it to the user’s language.
- If no relevant information exists, say exactly:
  - Turkish: "Bu konuda elimde net bir bilgi bulunamadı."
  - English: "I couldn't find clear information about this."

PRIORITY:
1. Use context
2. Translate if necessary
3. Answer in user's language

Important: The user's question language is {answer_language_instruction}. Your final answer must be in {answer_language_instruction}.

Bağlam:
{context}

Kullanıcı sorusu:
{question}
 
 Cevap (yalnızca bağlama dayanarak):
"""
        answer = ask_gemma(prompt)
        if answer.startswith("Gemma error:"):
            return _persist_assistant_reply(conv, answer, status=502, as_detail=True)
        if not (answer or "").strip():
            return _persist_assistant_reply(conv, no_info_msg)
        # Enforce answer language: if model drifts (or mixes languages), translate back
        # without adding facts. Be strict for EN questions (UI expectation).
        if is_tr:
            if (not _looks_turkish(answer)) or _looks_english(answer):
                answer = _translate_answer(answer, "tr")
        else:
            if (not _looks_english(answer)) or _looks_turkish(answer):
                answer = _translate_answer(answer, "en")
        if address_intent:
            answer = _strip_urls_plain_text(answer)
        return _persist_assistant_reply(conv, answer)
    except Exception:
        logger.exception("Failed to answer question in /ask")
        err_text = "Backend initialization failed. Check model/dependencies and server logs."
        return _persist_assistant_reply(conv, err_text, status=500, as_detail=True)


@csrf_exempt
@require_http_methods(["GET", "POST"])
def conversations_root(request):
    from chatbot.models import Conversation

    if request.method == "GET":
        qs = Conversation.objects.all().order_by("-updated_at")[:200]
        results = [
            {
                "id": c.id,
                "title": c.title or "",
                "created_at": c.created_at.isoformat(),
                "updated_at": c.updated_at.isoformat(),
            }
            for c in qs
        ]
        return JsonResponse({"results": results})
    conv = Conversation.objects.create(title="")
    return JsonResponse(
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
        },
        status=201,
    )


@csrf_exempt
@require_http_methods(["GET"])
def conversations_detail(request, pk):
    from chatbot.models import Conversation

    conv = Conversation.objects.filter(pk=pk).first()
    if not conv:
        return JsonResponse({"detail": "Not found."}, status=404)
    msgs = [
        {
            "id": m.id,
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat(),
        }
        for m in conv.messages.all()
    ]
    return JsonResponse(
        {
            "id": conv.id,
            "title": conv.title,
            "created_at": conv.created_at.isoformat(),
            "updated_at": conv.updated_at.isoformat(),
            "messages": msgs,
        }
    )
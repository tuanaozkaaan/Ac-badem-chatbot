import hashlib
import heapq
import json
import logging
import os
import time
import re
import unicodedata
import difflib
from urllib.parse import urlparse
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests
from django.db.models import Count, Max, Q
from django.http import JsonResponse
from django.utils import timezone
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

# Crawled chunks often omit the postal block; official page:
# https://acibadem.edu.tr/kayit/iletisim/ulasim
ACIBADEM_GENERAL_FOCUS_BLOCK = (
    "[Genel tanıtım — odak]\n"
    "Acıbadem Üniversitesi kökleri itibarıyla sağlık bilimleri, tıp, hemşirelik, eczacılık ve benzeri alanlarda "
    "güçlü bir vakıf üniversitesidir; mühendislik ve diğer fakülteler program yelpazesinin parçasıdır. "
    "Genel 'üniversite nedir / kısa bilgi' sorularında tek bir mühendislik bölümünü veya tek kişiyi "
    "üniversitenin ana kimliği gibi sunma; önce sağlık ve çok disiplinli yapıyı özetle.\n"
)

OFFICIAL_CAMPUS_ADDRESS_BLOCK = (
    "[Resmî kampüs adresi ve iletişim — Acıbadem Üniversitesi; "
    "kaynak: acibadem.edu.tr/kayit/iletisim/ulasim]\n"
    "Kerem Aydınlar Kampüsü, Kayışdağı Cad. No:32, 34752 Ataşehir/İstanbul\n"
    "Telefon: 0216 500 44 44, 0216 576 50 76\n"
    "E-posta: info@acibadem.edu.tr\n"
)

_CE_OVERVIEW_FALLBACK = (
    "[Özet kaynak — Bilgisayar Mühendisliği lisans; resmî müfredat değildir]\n"
    "Bilgisayar Mühendisliği lisans programı yazılım ve donanımı birlikte ele alır; algoritma, veri yapıları, "
    "işletim sistemleri ve mimari, veritabanları, ağlar, yazılım mühendisliği ve proje dersleri tipik kapsamdadır "
    "(ders adları üniversiteye göre değişir). Bilgisayar Programcılığı önlisans programı ayrı bir düzeydedir."
)

_ENGINEERING_DEPARTMENTS_FILE = "engineering_natural_sciences_departments.txt"
_ENGINEERING_DEPARTMENTS_FALLBACK = (
    "Mühendislik ve Doğa Bilimleri Fakültesi şu bölümleri içerir:\n"
    "- Bilgisayar Mühendisliği\n"
    "- Biyomedikal Mühendisliği\n"
    "- Moleküler Biyoloji ve Genetik (MBG)\n\n"
    "Not: Bu bilgi yerel veri dosyalarından derlenmiştir."
)
_SAFE_FALLBACK_TR = (
    "Bu bilgi yerel veri kaynaklarında net olarak bulunamadı. "
    "En doğru ve güncel bilgi için Acıbadem Üniversitesi’nin resmi web sitesini kontrol etmeniz önerilir."
)
_SAFE_FALLBACK_EN = (
    "This information was not clearly found in the local data sources. "
    "For the most accurate and up-to-date information, please check Acıbadem University’s official website."
)


def _ce_overview_context_block() -> str:
    """Short CE overview from repo data/ (Docker volume); fallback if file missing."""
    p = Path(__file__).resolve().parent.parent / "data" / "bilgisayar_muhendisligi.txt"
    try:
        if p.is_file():
            raw = p.read_text(encoding="utf-8", errors="replace").strip()
            if raw:
                return f"[Özet kaynak — Bilgisayar Mühendisliği lisans; resmî müfredat değildir]\n{raw}"
    except OSError:
        pass
    return _CE_OVERVIEW_FALLBACK


def _engineering_faculty_departments_intent(question: str) -> bool:
    q = _ascii_fold_turkish(question or "")
    has_faculty = "muhendislik ve doga bilimleri fakultesi" in q
    asks_for_list = any(
        n in q
        for n in (
            "hangi bolum",
            "bolumleri",
            "bolumler",
            "icerir",
            "nelerdir",
        )
    )
    return has_faculty and asks_for_list


def _engineering_faculty_departments_reply() -> str:
    p = Path(__file__).resolve().parent.parent / "data" / _ENGINEERING_DEPARTMENTS_FILE
    try:
        if p.is_file():
            lines = []
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line.startswith("- "):
                    item = line[2:].strip()
                    if item:
                        lines.append(item)
            if lines:
                unique: list[str] = []
                seen: set[str] = set()
                for item in lines:
                    key = item.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    unique.append(item)
                body = "\n".join(f"- {name}" for name in unique)
                return (
                    "Mühendislik ve Doğa Bilimleri Fakültesi şu bölümleri içerir:\n"
                    f"{body}\n\n"
                    "Not: Bu bilgi yerel veri dosyalarından derlenmiştir."
                )
    except OSError:
        pass
    return _ENGINEERING_DEPARTMENTS_FALLBACK


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


def _green_or_sustainable_campus_question(question: str) -> bool:
    """
    Sustainability / green campus topics. These must NOT trigger postal-address
    routing (word 'kampüs' alone would otherwise match address_intent).
    """
    raw = (question or "").lower()
    if any(
        x in raw
        for x in (
            "sürdürülebilir",
            "surdürülebilir",
            "yeşil kampüs",
            "yesil kampus",
            "çevre dostu",
            "iklim dostu",
            "karbon ayak",
            "karbon nötr",
            "leed",
            "eko kampüs",
            "eko kampus",
        )
    ):
        return True
    q = _ascii_fold_turkish(question or "")
    return any(
        x in q
        for x in (
            "surduurulebilir",
            "surdurulebilir",
            "sustainable",
            "green campus",
            "carbon neutral",
            "iklim",
            "leed",
            "eko kampus",
        )
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


def _answer_is_stock_no_info(answer: str) -> bool:
    al = re.sub(r"\s+", " ", (answer or "").strip().lower())
    return al.startswith("bu konuda elimde net bir bilgi bulunamad") or al.startswith(
        "i couldn't find clear information about this"
    )


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
    if long_hits >= 1 or short_hits >= 2:
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


def _split_context_blocks(context: str) -> list[str]:
    parts = [p.strip() for p in re.split(r"\n\s*---\s*\n", context or "") if p.strip()]
    return parts if parts else ([context.strip()] if (context or "").strip() else [])


def _detect_specific_faculty_focus(question: str) -> str | None:
    q = _ascii_fold_turkish(question or "")
    if "saglik bilimleri fakultesi" in q or "sağlık bilimleri fakültesi" in (question or "").lower():
        return "saglik_bilimleri"
    if "muhendislik ve doga bilimleri" in q or "mühendislik ve doğa bilimleri" in (question or "").lower():
        return "muhendislik_doga"
    return None


def _extract_faculty_phrase(question: str) -> str | None:
    q = _ascii_fold_turkish(question or "")
    known = (
        "saglik bilimleri fakultesi",
        "muhendislik ve doga bilimleri fakultesi",
        "tip fakultesi",
        "eczacilik fakultesi",
        "insan ve toplum bilimleri fakultesi",
    )
    for k in known:
        if k in q:
            return k
    return None


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


def _is_extractive_question(question: str) -> bool:
    q = _ascii_fold_turkish(question or "")
    cues = (
        "hangi bolumleri icerir",
        "hangi bolumler var",
        "hangi fakultede yer alir",
        "kimdir",
        "nerede",
        "iletisim bilgileri",
        "adres nedir",
    )
    return any(c in q for c in cues)


def _extractive_department_list(question: str, context: str) -> tuple[str, str] | None:
    q = _ascii_fold_turkish(question or "")
    if not any(x in q for x in ("hangi bolumleri icerir", "hangi bolumler var")):
        return None

    faculty_phrase = _extract_faculty_phrase(question)
    blocks = _split_context_blocks(context)
    if faculty_phrase:
        blocks = [b for b in blocks if faculty_phrase in _ascii_fold_turkish(b)]
    if not blocks:
        return None

    dept_candidates: list[str] = []
    other_faculty_markers = (
        "muhendislik ve doga bilimleri fakultesi",
        "saglik bilimleri fakultesi",
        "tip fakultesi",
        "eczacilik fakultesi",
        "insan ve toplum bilimleri fakultesi",
    )
    for block in blocks:
        for raw_line in block.splitlines():
            line = raw_line.strip(" -\t")
            lf = _ascii_fold_turkish(line)
            if not line:
                continue
            if line.startswith("Source URL:") or line.startswith("Page Title:"):
                continue
            if faculty_phrase:
                for marker in other_faculty_markers:
                    if marker != faculty_phrase and marker in lf:
                        line = ""
                        break
                if not line:
                    continue
            if any(k in lf for k in ("muhendisligi", "bolumu", "programi", "programı")):
                if len(line) <= 120:
                    dept_candidates.append(line)

    # preserve order, dedupe
    uniq: list[str] = []
    seen: set[str] = set()
    for d in dept_candidates:
        key = _ascii_fold_turkish(d)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(d)

    if not uniq:
        return None

    if faculty_phrase:
        pretty_faculty = {
            "saglik bilimleri fakultesi": "Sağlık Bilimleri Fakültesi",
            "muhendislik ve doga bilimleri fakultesi": "Mühendislik ve Doğa Bilimleri Fakültesi",
            "tip fakultesi": "Tıp Fakültesi",
            "eczacilik fakultesi": "Eczacılık Fakültesi",
            "insan ve toplum bilimleri fakultesi": "İnsan ve Toplum Bilimleri Fakültesi",
        }.get(faculty_phrase, "İlgili Fakülte")
        header = f"{pretty_faculty} şu bölümleri içerir:"
    else:
        header = "İlgili fakülte şu bölümleri içerir:"
    body = "\n".join(f"- {d}" for d in uniq[:12])
    return f"{header}\n{body}", "department_list"


def _extractive_person_or_title(question: str, context: str) -> tuple[str, str] | None:
    q = question or ""
    q_fold = _ascii_fold_turkish(q)
    if "kimdir" not in q_fold:
        return None
    # simple name capture: words starting with uppercase letters
    m = re.findall(r"\b[A-ZÇĞİÖŞÜ][a-zçğıöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+){1,3}\b", q)
    if not m:
        return None
    name = m[-1].strip()
    nf = _ascii_fold_turkish(name)
    title_keys = ("dekan", "bolum baskani", "bölüm başkanı", "prof", "doc", "dr", "chair", "head")
    for sentence in re.split(r"[\n\.]+", context or ""):
        s = sentence.strip()
        sf = _ascii_fold_turkish(s)
        if nf in sf and any(k in sf for k in title_keys):
            return s + ".", "person_title"
    return None


def _extractive_contact_or_address(question: str, context: str) -> tuple[str, str] | None:
    qf = _ascii_fold_turkish(question or "")
    asks_address = "adres" in qf or "nerede" in qf
    asks_contact = "iletisim" in qf or "telefon" in qf or "email" in qf or "e-posta" in qf
    if not asks_address and not asks_contact:
        return None

    lines = [ln.strip() for ln in (context or "").splitlines() if ln.strip()]
    picked: list[str] = []
    for ln in lines:
        lf = _ascii_fold_turkish(ln)
        if ln.startswith("Source URL:") or ln.startswith("Page Title:"):
            continue
        if asks_address and any(k in lf for k in ("kampus", "cad", "no:", "atasehir", "istanbul", "adres", "kayisdagi")):
            picked.append(ln)
        if asks_contact and any(k in lf for k in ("telefon", "e-posta", "email", "info@acibadem.edu.tr")):
            picked.append(ln)
    uniq = []
    seen = set()
    for p in picked:
        k = _ascii_fold_turkish(p)
        if k in seen:
            continue
        seen.add(k)
        uniq.append(p)
    # Keep only official-like contact lines to avoid random person/bank rows.
    official = [
        u for u in uniq if any(k in _ascii_fold_turkish(u) for k in ("acibadem.edu.tr", "info@acibadem.edu.tr", "telefon", "kampus", "kayisdagi", "atasehir"))
    ]
    uniq = official or uniq
    if asks_contact and not any("acibadem.edu.tr" in _ascii_fold_turkish(u) or "info@acibadem.edu.tr" in _ascii_fold_turkish(u) for u in uniq):
        return None
    if not uniq:
        return None
    return "\n".join(uniq[:8]), "contact_or_address"


def _try_extractive_answer(question: str, context: str) -> tuple[str, str] | None:
    for fn in (_extractive_department_list, _extractive_person_or_title, _extractive_contact_or_address):
        out = fn(question, context)
        if out:
            return out
    return None


def _select_context_for_llm(question: str, context: str, *, max_chunks: int = 5, max_chars: int = 5000) -> tuple[str, list[str], int]:
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
    tf = _ascii_fold_turkish(t)
    # Model çıktısı sıkça ASCII (i̇→i); tek " ve " ile kaçırmayıp gereksiz çeviri / ikinci Ollama çağrısı tetiklenmesin.
    strong = (
        " ogrenci ",
        "ogrenci ",
        " universite",
        "universitesi",
        " fakulte",
        " bolum",
        " iletisim",
        " telefon",
        " e-posta",
        " eposta",
        " adres",
        " kampus",
        " istanbul",
        " turkiye",
        " icin ",
        " bilgi ",
        " basvuru",
        " kayit",
    )
    if any(s in tf for s in strong):
        return True
    tr_words = (" ve ", " ile ", " icin ", " adres", " universite", " kampus", " istanbul", " turkiye", " nedir", " nerede")
    hits = sum(1 for w in tr_words if w in f" {tf} ")
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


def _cs_engineering_lisans_intent(question: str) -> bool:
    """
    User asks about Computer *Engineering* (lisans), not the separate associate
    'Bilgisayar Programcılığı' (önlisans) program.
    """
    q = _ascii_fold_turkish(question or "")
    if "computer engineering" in q:
        return True
    if "bilgisayar" not in q:
        return False
    if "programcilik" in q and "muhendislik" not in q and "muhendisligi" not in q:
        return False
    return "muhendisligi" in q or "muhendislik" in q


def _cs_engineering_course_catalog_intent(question: str) -> bool:
    """
    Kullanıcı somut ders listesi / sınıf / müfredat / kod istiyor; genel 'alan tanımı' değil.
    Bu durumda repo özet dosyası (ce_block) enjekte edilmemeli — yalnızca DB/OBS chunk'ları.
    """
    if not _cs_engineering_lisans_intent(question):
        return False
    q = _ascii_fold_turkish(question or "")
    needles = (
        "1. sinif",
        "2. sinif",
        "3. sinif",
        "4. sinif",
        "birinci sinif",
        "ikinci sinif",
        "ucuncu sinif",
        "dorduncu sinif",
        "sinif ders",
        "siniftaki ders",
        "ders list",
        "derslerin",
        "dersleri",
        "dersler",
        "hangi ders",
        "hangi dersler",
        "dersler var",
        "programinda hangi",
        "programda hangi",
        "mufredat",
        "müfredat",
        "katalog",
        "curriculum",
        "syllabus",
        "ders kodu",
        "ders kod",
        "kredi",
        "akts",
        "yariyil",
        "yarıyıl",
        "donem",
        "dönem",
        "bologna",
        "program ciktisi",
        "program çıktısı",
        "ogrenme ciktisi",
        "öğrenme çıktısı",
    )
    return any(n in q for n in needles)


def _asks_subunits_of_named_faculty(question: str) -> bool:
    """
    Tek bir fakülte/yüksekokul adı verilmiş ve altındaki bölüm/program soruluyorsa True.
    Bu durumda _faculty_department_catalog_intent açılmamalı — yoksa tüm üniversite OR araması
    ve ağır dept_catalog yolu tetiklenir (dakikalarca sürebilir, bağlam da dağılır).
    """
    qf = _ascii_fold_turkish(question or "")
    if "fakultes" not in qf:
        return False
    if not any(
        x in qf
        for x in (
            "hangi bolum",
            "hangi program",
            "nelerdir",
            "icerir",
            "hangi lisans",
            "hangi onlisans",
        )
    ):
        return False
    units = (
        "muhendislik ve doga",
        "tip fakulte",
        "eczacilik fakulte",
        "saglik bilimleri fakulte",
        "guzel sanatlar",
        "hukuk fakulte",
        "iletisim fakulte",
        "dis hekimligi",
        "egitim bilimleri fakulte",
        "fen edebiyat",
        "yabanci diller",
    )
    return any(u in qf for u in units)


def _general_acibadem_intro_intent(question: str) -> bool:
    """
    'Üniversite hakkında kısa bilgi' gibi genel tanıtım — BM/tek programa kilitlenmesin diye retrieval + prompt ayarı.
    """
    qf = _ascii_fold_turkish(question or "")
    if "acibadem" not in qf and "aci badem" not in qf:
        return False
    if _cs_engineering_lisans_intent(question) or _cs_engineering_course_catalog_intent(question):
        return False
    if "bilgisayar" in qf or "computer" in qf or "muhendisligi" in qf:
        return False
    hints = (
        "hakkinda",
        "bilgi ver",
        "kisaca",
        "kisa bilgi",
        "nedir",
        "tanit",
        "tanitir misin",
        "genel bilgi",
        "universiteyi anlat",
        "universite hakkinda",
        "universitesi hakkinda",
        "hakkinda bilgi",
    )
    return any(h in qf for h in hints)


def _faculty_department_catalog_intent(question: str) -> bool:
    """Broad questions asking which faculties/departments/programs exist at the university."""
    ql = (question or "").lower()
    qf = _ascii_fold_turkish(question or "")
    if _asks_subunits_of_named_faculty(question):
        return False
    needles = (
        "hangi bölüm",
        "hangi bolum",
        "bölümler",
        "bolumler",
        "tüm bölüm",
        "tum bolum",
        "hangi fakülte",
        "hangi fakulte",
        "fakülteler",
        "fakulteler",
        "hangi program",
        "hangi programlar",
        "bölüm var",
        "bolum var",
        "fakülte var",
        "fakulte var",
    )
    if any(n in ql for n in needles):
        return True
    if any(x in ql for x in ("departments", "faculties", "schools at")):
        return True
    if "hangi" in qf and ("bolum" in qf or "fakulte" in qf or "program" in qf):
        return True
    # "Eczacılık fakültesini anlat" gibi tek fakülte sayfası — eskisi gibi geniş bağlam + çoklu URL.
    if "fakultes" in qf and any(
        x in qf for x in ("anlat", "tanit", "acikla", "nedir", "hakkinda", "bilgi ver")
    ):
        return True
    return False


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
    `RETRIEVE_MAX_CANDIDATES` (default 1000). If no keyword matches, the most recently updated
    chunks up to that cap are used instead.
    """
    keywords = _extract_keywords(question)
    # Question words widen OR-matches without helping entity questions ("kimdir" on half the site).
    lookup_noise = frozenset(
        {"kimdir", "nedir", "midir", "nerede", "nasıl", "nasil", "hangi", "neden", "niçin", "nicin", "kim", "ne"}
    )
    keywords = [k for k in keywords if k not in lookup_noise]
    q_lower = (question or "").lower()
    cs_eng_intent = _cs_engineering_lisans_intent(question)
    course_catalog_intent = _cs_engineering_course_catalog_intent(question)
    green_campus_q = _green_or_sustainable_campus_question(question)
    dept_catalog_intent = _faculty_department_catalog_intent(question)
    general_intro_intent = _general_acibadem_intro_intent(question)

    # Multi-word phrases (not produced by whitespace tokenization) improve DB icontains narrowing.
    extra_lookup_terms: list[str] = []
    if "bilgisayar mühendisliği" in q_lower or "bilgisayar muhendisligi" in q_lower:
        extra_lookup_terms.extend(
            ["bilgisayar mühendisliği", "bilgisayar muhendisligi", "bilgisayar mühendis", "bilgisayar muhendis"]
        )
    if cs_eng_intent:
        extra_lookup_terms.extend(
            [
                "mühendislik fakültesi",
                "muhendislik fakultesi",
                "computer engineering",
                "lisans programı",
                "lisans programi",
            ]
        )
    if course_catalog_intent:
        extra_lookup_terms.extend(
            [
                "obs.acibadem",
                "obs.acibadem.edu.tr",
                "bologna",
                "müfredat",
                "mufredat",
                "ders bilgi",
                "ders kodu",
                "akts",
                "kredi",
                "yarıyıl",
                "yariyil",
                "güz",
                "guz",
                "bahar",
                "bilgisayar mühendisliği",
                "bilgisayar muhendisligi",
                "öğrenim",
                "ogrenim",
            ]
        )
    if green_campus_q:
        extra_lookup_terms.extend(
            [
                "sürdürülebilir",
                "surdurulebilir",
                "sustainable",
                "yeşil kampüs",
                "yesil kampus",
                "iklim",
                "çevre",
                "karbon",
            ]
        )
    if "bölüm başkanı" in q_lower or "bolum baskani" in q_lower:
        extra_lookup_terms.extend(["bölüm başkanı", "bolum baskani", "bölüm başkan", "bolum baskan"])
    if dept_catalog_intent:
        extra_lookup_terms.extend(
            [
                "fakülte",
                "fakulte",
                "fakülteler",
                "fakulteler",
                "mühendislik",
                "muhendislik",
                "tıp fakültesi",
                "tip fakultesi",
                "eczacılık",
                "hemşirelik",
                "beslenme",
                "fizyoterapi",
                "sağlık bilimleri",
                "saglik bilimleri",
                "mühendislik ve doğa",
                "muhendislik ve doga",
                "güzel sanatlar",
                "guzel sanatlar",
                "hukuk fakültesi",
                "hukuk fakultesi",
                "diş hekimliği",
                "dis hekimligi",
                "üniversitemiz",
                "universitemiz",
                "lisans",
                "önlisans",
                "onlisans",
            ]
        )
    if _asks_subunits_of_named_faculty(question):
        qfx = _ascii_fold_turkish(question or "")
        if "muhendislik ve doga" in qfx:
            extra_lookup_terms.extend(
                [
                    "mühendislik ve doğa bilimleri",
                    "muhendislik ve doga bilimleri",
                    "muhendislik ve doga",
                ]
            )
        if "tip fakulte" in qfx:
            extra_lookup_terms.extend(["tıp fakültesi", "tip fakultesi"])
        if "eczacilik fakulte" in qfx:
            extra_lookup_terms.extend(["eczacılık fakültesi", "eczacilik fakultesi"])
        if "saglik bilimleri fakulte" in qfx:
            extra_lookup_terms.extend(["sağlık bilimleri", "saglik bilimleri"])
    if general_intro_intent:
        extra_lookup_terms.extend(
            [
                "sağlık bilimleri",
                "saglik bilimleri",
                "tıp fakültesi",
                "tip fakultesi",
                "vakıf üniversitesi",
                "vakif universitesi",
                "Acıbadem Sağlık",
                "hemşirelik",
                "hemsirelik",
                "eczacılık",
                "kuruluş",
                "kurulus",
            ]
        )

    # Intent detection (priority matters):
    # - If question mentions staj/internship => internship intent
    # - Else if question mentions apply/admission terms => general admission intent
    internship_intent = any(t in q_lower for t in ("staj", "internship"))
    admission_intent = (not internship_intent) and any(
        t in q_lower for t in ("başvuru", "basvuru", "admission", "apply", "application")
    )
    address_intent = (not green_campus_q) and any(
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

    # Uzun sayfa metinlerinde her satırda tüm gövdeyi skorlamak CPU'da onlarca sn sürebilir; eşleştirme için önek yeter.
    _score_body_limit = int(os.environ.get("RETRIEVE_CHUNK_SCORE_CHARS", "5500"))
    _score_body_limit = max(2800, min(_score_body_limit, 120_000))

    def _score_row(row) -> int:
        title = (row.title or "").lower()
        section = (row.section or "").lower()
        raw_body = row.chunk_text or ""
        if len(raw_body) > _score_body_limit:
            raw_body = raw_body[:_score_body_limit]
        text = raw_body.lower()
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
            # Do not treat associate "programcılığı" pages as the engineering department chair source.
            if "bilgisayar-programciligi" in url or "bilgisayar-programcılığı" in url:
                score -= 180
            if any(x in url for x in ("muhendislik", "mühendislik", "engineering")):
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

        if cs_eng_intent:
            u = url.lower()
            if "bilgisayar-programciligi" in u or "bilgisayar-programcılığı" in u:
                score -= 280
            if "onlisans" in u or "onlisans" in text:
                score -= 220
            if ("programcılık" in text or "programcilik" in text) and "muhendislik" not in text and "mühendislik" not in text:
                score -= 120
            if any(x in u for x in ("muhendislik", "mühendislik", "engineering", "lisans")):
                score += 100
            if "bilgisayar mühendisliği" in text or "bilgisayar muhendisligi" in text:
                score += 120
            if "mühendislik fakültesi" in text or "muhendislik fakultesi" in text:
                score += 90

        if green_campus_q:
            blob = f"{text} {title} {section}".lower()
            if any(
                x in blob
                for x in (
                    "sürdürülebilir",
                    "surdurulebilir",
                    "sustainable",
                    "yeşil",
                    "yesil",
                    "iklim",
                    "karbon",
                    "leed",
                    "çevre",
                    "eko",
                )
            ):
                score += 110
            st = (section or "").lower()
            if st == "contact_address" and not any(
                x in blob for x in ("sürdürülebilir", "surdurulebilir", "sustainable", "iklim", "karbon", "leed")
            ):
                score -= 150

        if dept_catalog_intent:
            tl = (title or "").lower()
            sl = (section or "").lower()
            if "fakülte" in tl or "fakulte" in tl or "fakülte" in sl or "fakulte" in sl:
                score += 70
            if sum(1 for w in ("tıp", "eczacılık", "mühendislik", "hemşirelik", "beslenme", "fizyoterapi") if w in text) >= 2:
                score += 45
            if "/akademik/" in url and ("lisans" in url or "onlisans" in url or "yuksekokul" in url):
                score += 35
            uroot = url.rstrip("/").lower()
            if uroot.endswith("acibadem.edu.tr") and any(
                x in text for x in ("fakülte", "fakulte", "tıp", "mühendislik", "eczacılık", "hemşirelik")
            ):
                score += 115
            # Tam fakülte listesinde çok sayfa var; tek fakülte URL'sine aşırı boost tüm sırayı eczacılıkta topluyor.
            if "eczacilik" in url or "eczacılık" in url:
                score += 28
            blob_fac = _ascii_fold_turkish(f"{title} {section} {text}")
            fr = _faculty_richness(blob_fac)
            if fr >= 70:
                score += min(220, fr + 40)
            elif fr >= 42:
                score += 85
            elif fr >= 14:
                score += 28

        if general_intro_intent:
            blob_gi = _ascii_fold_turkish(f"{title} {section} {text} {url}")
            u = url.lower()
            if any(
                x in u
                for x in (
                    "bilgisayar-muhendisligi",
                    "bilgisayar_muhendisligi",
                    "computer-engineering",
                )
            ):
                score -= 280
            if "bilgisayar" in title and "muhendis" in _ascii_fold_turkish(title):
                score -= 160
            if any(
                x in blob_gi
                for x in (
                    "saglik bilimleri",
                    "tip fakulte",
                    "tip fakültesi",
                    "eczacilik",
                    "hemsirelik",
                    "dis hekimligi",
                    "vakif universitesi",
                    "kurulus",
                    "saglik grubu",
                    "hastane",
                )
            ):
                score += 120

        if course_catalog_intent:
            st = (getattr(row, "source_type", None) or "").lower()
            if st == "obs":
                score += 110
            if "obs.acibadem" in url:
                score += 130
            blob_ct = f"{text} {title}"
            for kw in (
                "müfredat",
                "mufredat",
                "bologna",
                "akts",
                "kredi",
                "yarıyıl",
                "yariyil",
                "güz",
                "guz",
                "bahar",
                "ders kodu",
                "ders kod",
                "öğrenim",
                "ogrenim",
                "program çıktısı",
                "program ciktisi",
            ):
                if kw in blob_ct:
                    score += 38
            if re.search(r"\b[a-zçğıöşü]{2,5}\s*\d{3}\b", blob_ct, re.I):
                score += 72

        return score

    # 1) Prefer DB chunks if available — score a bounded candidate set, then take top k by score.
    try:
        from chatbot.models import PageChunk

        if PageChunk.objects.exists():
            # 6000 satırı tamamen RAM'de skorlamak yavaş; varsayılanı düşük tut (env ile artırılabilir).
            max_candidates = max(1, int(os.environ.get("RETRIEVE_MAX_CANDIDATES", "1000")))
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

            def _iter_scored_rows():
                for row in candidate_qs.iterator(chunk_size=400):
                    yield (_score_row(row), str(row.updated_at or ""), row)

            pool_size = max(
                1,
                min(
                    max_candidates,
                    max(500, k * 28) if dept_catalog_intent else max(280, k * 22),
                ),
            )
            scored_pool = heapq.nlargest(
                pool_size,
                _iter_scored_rows(),
                key=lambda t: (t[0], t[1]),
            )
            # Düşük veya negatif skorlarda bile en iyi k adayı kullan: erken return "" hem RAG'ı
            # keser hem de aşağıdaki data/*.txt yedeğine düşmeyi engeller (yorumla uyumlu).

            logger.info("retrieve_context(db): question=%r pool=%s (max_cand=%s)", question, len(scored_pool), max_candidates)
            for rank, (s, _u, row) in enumerate(scored_pool[:k], start=1):
                logger.info("retrieve_context(db): #%s score=%s title=%r", rank, s, row.title)

            if dept_catalog_intent:
                picked: list = []
                url_counts: dict[str, int] = {}
                seen_pk: set[int] = set()
                # Liste sorusunda farklı URL'lerden parça al — aynı sitede 2 parça yerine 14 ayrı kaynak.
                max_per_url = 1

                def _take_row(row) -> bool:
                    pk = int(row.pk)
                    if pk in seen_pk:
                        return False
                    urlv = ((row.url or "")[:220] or "?").lower()
                    if url_counts.get(urlv, 0) >= max_per_url:
                        return False
                    picked.append(row)
                    seen_pk.add(pk)
                    url_counts[urlv] = url_counts.get(urlv, 0) + 1
                    return True

                for _s, _u, row in scored_pool:
                    if len(picked) >= k:
                        break
                    _take_row(row)
                for _s, _u, row in scored_pool:
                    if len(picked) >= k:
                        break
                    if int(row.pk) in seen_pk:
                        continue
                    picked.append(row)
                    seen_pk.add(int(row.pk))
                # İlk eşleşen parça sık sık yalnızca 1–2 fakülte içerir; en zengin "genel liste" parçasını seç.
                overview_row = None
                overview_score = -1
                overview_tuple: tuple[int, str, object] | None = None
                for _s, _u, row in scored_pool[:280]:
                    uu = (row.url or "").rstrip("/").lower()
                    if not uu.endswith("acibadem.edu.tr"):
                        continue
                    tx = (row.chunk_text or "") + " " + (row.title or "")
                    fr = _faculty_richness(_ascii_fold_turkish(tx.lower()))
                    # ~3+ farklı birim ipucu (≈42+) olan parça genelde tam liste / akademik özet sayfasıdır.
                    if fr < 42:
                        continue
                    if fr > overview_score or (
                        fr == overview_score and overview_tuple is not None and _s > overview_tuple[0]
                    ):
                        overview_score = fr
                        overview_tuple = (_s, _u, row)
                if overview_tuple is not None:
                    overview_row = overview_tuple[2]
                if overview_row is not None and int(overview_row.pk) not in {int(r.pk) for r in picked}:
                    picked = [overview_row] + picked[: max(0, k - 1)]
                top_rows = picked[:k]
            else:
                top_rows = [row for _, _, row in scored_pool[:k]]
            blocks: list[str] = []
            for c in top_rows:
                meta = " | ".join(
                    [p for p in [c.title or "", c.section or "", c.source_type or "", c.url or ""] if p]
                )
                blocks.append(f"[{meta}]\n{c.chunk_text}" if meta else c.chunk_text)

            merged_db = "\n\n---\n\n".join(blocks).strip()
            if merged_db:
                return merged_db
    except Exception:
        logger.exception("retrieve_context: DB veya skorlama hatası (yedeğe düşülüyor)")

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


_STRICT_RAG_NOT_FOUND = "BİLGİ BULUNAMADI (NO CONTEXT FROM DB)"


@lru_cache(maxsize=1)
def _sentence_transformer_for_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def _embed_query_normalized(question: str, model_name: str) -> np.ndarray:
    """Single-query embedding, L2-normalized (cosine similarity via dot product)."""
    model = _sentence_transformer_for_model(model_name)
    v = model.encode([question], convert_to_numpy=True, normalize_embeddings=True)
    out = np.asarray(v[0], dtype=np.float32).ravel()
    n = float(np.linalg.norm(out) + 1e-12)
    return out / n


@lru_cache(maxsize=16)
def _embedding_matrix_pack(cache_key: tuple[str, int, int]) -> tuple[np.ndarray, tuple[dict[str, str | int], ...]]:
    """
    Tüm embedding satırlarını tek seferde matrise çevirir; cache_key (kaynak filtresi + satır sayısı + max id)
    değişene kadar @lru_cache ile bellekte tutulur — her /ask isteğinde 1947 kez JSON parse etmez.
    """
    from chatbot.models import ChunkEmbedding
    from rag.document_loader import EXPECTED_EMBEDDING_MODEL

    kind, _, __ = cache_key
    source_type = None if kind == "__all__" else kind

    qs = ChunkEmbedding.objects.select_related("chunk")
    if source_type:
        qs = qs.filter(chunk__source_type=source_type)

    vectors: list[np.ndarray] = []
    metas: list[dict[str, str | int]] = []

    for emb in qs.iterator(chunk_size=800):
        ch = emb.chunk
        if not ch:
            continue
        text = (ch.chunk_text or "").strip()
        if not text:
            continue
        if emb.vector is None:
            continue
        name = (emb.embedding_model or "").strip()
        if name and name != EXPECTED_EMBEDDING_MODEL:
            continue
        arr = np.asarray(emb.vector, dtype=np.float32).ravel()
        if arr.size == 0:
            continue
        nrm = float(np.linalg.norm(arr) + 1e-12)
        vn = arr / nrm
        vectors.append(vn.astype(np.float32, copy=False))
        metas.append(
            {
                "chunk_id": int(ch.pk),
                "url": (ch.url or "").strip(),
                "title": (ch.title or "").strip(),
                "text": text,
            }
        )

    if not vectors:
        return np.zeros((0, 0), dtype=np.float32), ()

    mat = np.stack(vectors, axis=0)
    return mat, tuple(metas)


def _retrieve_top_chunks_by_embedding(
    question: str,
    k: int,
    *,
    source_type: str | None = None,
) -> list[dict]:
    """
    Read-only retrieval: ChunkEmbedding.vector + PageChunk.chunk_text.
    source_type='obs' ile yalnızca OBS chunk'ları taranır (ders kataloğu için çok daha hızlı).
    """
    from chatbot.models import ChunkEmbedding
    from rag.document_loader import EXPECTED_EMBEDDING_MODEL

    base = ChunkEmbedding.objects.select_related("chunk")
    if source_type:
        base = base.filter(chunk__source_type=source_type)
    sig = base.aggregate(c=Count("id"), mx=Max("id"))
    cache_key = (source_type or "__all__", int(sig["c"] or 0), int(sig["mx"] or 0))

    mat, metas_t = _embedding_matrix_pack(cache_key)
    metas = list(metas_t)
    if mat.shape[0] == 0 or mat.shape[1] == 0 or not metas:
        return []

    qv = _embed_query_normalized(question, EXPECTED_EMBEDDING_MODEL)
    if int(qv.shape[0]) != mat.shape[1]:
        logger.warning(
            "embedding_dim_mismatch q_dim=%s mat_dim=%s — boş dönülüyor",
            qv.shape[0],
            mat.shape[1],
        )
        return []

    sims = mat @ qv
    order = np.argsort(-sims)
    top_idx = order[: max(1, min(int(k), len(order)))]

    out: list[dict] = []
    for i in top_idx:
        base_row = metas[int(i)]
        row = {
            "chunk_id": base_row["chunk_id"],
            "url": base_row["url"],
            "title": base_row["title"],
            "text": base_row["text"],
            "score": float(sims[int(i)]),
        }
        out.append(row)
    return out


def _strict_rag_verify_response(question: str, body: dict) -> JsonResponse:
    """
    Verification-only path: embedding retrieval from DB, no LLM, no Message/Conversation writes.
    """
    raw_k = body.get("strict_rag_top_k") or os.environ.get("ACU_STRICT_RAG_TOP_K") or 8
    try:
        k = int(raw_k)
    except (TypeError, ValueError):
        k = 8
    k = max(1, min(k, 50))

    chunks = _retrieve_top_chunks_by_embedding(question, k=k)
    if not chunks:
        return JsonResponse(
            {
                "strict_rag_verify": True,
                "answer": _STRICT_RAG_NOT_FOUND,
                "conversation_id": None,
                "retrieved_chunks": [],
            },
            status=200,
        )

    retrieved_lines: list[str] = ["[RETRIEVED CHUNKS]"]
    for i, c in enumerate(chunks, start=1):
        retrieved_lines.append(f"<chunk {i}>")
        retrieved_lines.append(f"score={c['score']:.4f}")
        retrieved_lines.append(f"url={c['url']}")
        retrieved_lines.append(f"title={c['title']}")
        retrieved_lines.append(c["text"])
        retrieved_lines.append("")

    max_chars = int(os.environ.get("ACU_STRICT_RAG_ANSWER_MAX_CHARS", "6000"))
    buf: list[str] = []
    rem = max_chars
    for c in chunks:
        if rem <= 0:
            break
        t = c["text"].strip()
        piece = t[:rem]
        buf.append(piece)
        rem -= len(piece)

    answer_only = "\n\n".join(buf)
    full_answer = "\n".join(retrieved_lines + ["[ANSWER BASED ON CONTEXT ONLY]", answer_only])

    slim = [
        {
            "chunk_id": c["chunk_id"],
            "score": c["score"],
            "url": c["url"],
            "title": c["title"],
            "text": c["text"],
        }
        for c in chunks
    ]

    return JsonResponse(
        {
            "strict_rag_verify": True,
            "answer": full_answer,
            "conversation_id": None,
            "retrieved_chunks": slim,
        },
        status=200,
    )


def ask_gemma(prompt: str) -> str:
    # CPU'da ilk üretim + uzun prompt 120s'yi aşabiliyor; varsayılanı yükselt (env ile düşürülebilir).
    ollama_http_timeout = int(os.environ.get("OLLAMA_HTTP_TIMEOUT", "90"))
    ollama_http_timeout = max(45, min(ollama_http_timeout, 900))
    try:
        base_url = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        model = (os.environ.get("OLLAMA_MODEL") or "gemma2:2b").strip()
        # Uzun liste/müfredat için yüksek tutulabilir; .env: OLLAMA_NUM_PREDICT (üst sınır 2048).
        raw_np = int(os.environ.get("OLLAMA_NUM_PREDICT", "140"))
        num_predict = max(64, min(raw_np, 2048))
        temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
        top_p = float(os.environ.get("OLLAMA_TOP_P", "0.8"))
        # Keep model loaded in VRAM/RAM between requests (e.g. "10m", "0" to unload). Empty = omit.
        keep_alive = (os.environ.get("OLLAMA_KEEP_ALIVE") or "").strip()
        payload: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_predict": num_predict,
                "temperature": temperature,
                "top_p": top_p,
            },
        }
        if keep_alive:
            payload["keep_alive"] = keep_alive
        # (bağlantı, okuma): yavaş üretimde okuma süresi env ile kontrol edilir.
        response = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=(min(30, ollama_http_timeout // 3), ollama_http_timeout),
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
    except requests.Timeout:
        return "__OLLAMA_TIMEOUT__"
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


def _append_followup_invite(text: str, *, is_tr: bool, conv_id: int, question: str) -> str:
    """Short closing line suggesting what the user might ask next (rotates per thread/question)."""
    t = (text or "").strip()
    if not t:
        return t
    salt = f"{conv_id}|{question[:96]}|{len(t)}"
    h = int(hashlib.sha256(salt.encode("utf-8", errors="replace")).hexdigest(), 16)
    if is_tr:
        variants = [
            "\n\n— Başvuru, burs veya kampüs için de sorabilirsin.",
            "\n\n— Program, staj veya iletişim hakkında devam edebilirsin.",
            "\n\n— Akademik takvim veya ulaşım için de sorabilirsin.",
            "\n\n— Kayıt veya yurt konularında da sorabilirsin.",
        ]
    else:
        variants = [
            "\n\n— Ask about admissions, scholarships, or campus if you like.",
            "\n\n— Programs, internships, or contact — happy to help.",
            "\n\n— Academic calendar or transport — just ask.",
        ]
    return t + variants[h % len(variants)]


def _persist_assistant_reply(
    conv,
    text: str,
    *,
    status: int = 200,
    as_detail: bool = False,
    attach_followup: bool = False,
    is_tr: bool = True,
    question: str = "",
) -> JsonResponse:
    from chatbot.models import Message

    if attach_followup and status == 200 and not as_detail:
        text = _append_followup_invite(text, is_tr=is_tr, conv_id=int(conv.pk), question=question or "")
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

    strict_rag = bool(body.get("strict_rag_verify")) or (
        (os.environ.get("ACU_STRICT_RAG_VERIFY") or "").strip().lower() in ("1", "true", "yes")
    )
    if strict_rag:
        return _strict_rag_verify_response(question, body)

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

        if _engineering_faculty_departments_intent(question):
            return _persist_assistant_reply(
                conv,
                _engineering_faculty_departments_reply(),
                attach_followup=False,
                is_tr=True,
                question=question,
            )

        # ASCII-fold so Turkish chars and .lower() quirks cannot skip the postal shortcut.
        q_fold = _ascii_fold_turkish(question)
        campus_green_q = _green_or_sustainable_campus_question(question)
        address_intent = (not campus_green_q) and any(
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
            return _persist_assistant_reply(
                conv,
                _canonical_campus_address_reply(is_tr),
                attach_followup=True,
                is_tr=is_tr,
                question=question,
            )

        cs_eng_q = _cs_engineering_lisans_intent(question)
        cs_course_catalog_q = _cs_engineering_course_catalog_intent(question)
        dept_cat = _faculty_department_catalog_intent(question)
        sub_fac_units = _asks_subunits_of_named_faculty(question)
        general_intro = _general_acibadem_intro_intent(question)
        k_ctx = 5
        if address_intent or cs_eng_q or campus_green_q:
            k_ctx = 8
        if general_intro:
            k_ctx = max(k_ctx, 7)
        if sub_fac_units:
            k_ctx = max(k_ctx, 10)
        if dept_cat:
            # Fakülte tam listesi için daha fazla parça + bağlam sınırı (model yine kısaltabilir).
            k_ctx = max(k_ctx, 18)
        if cs_course_catalog_q:
            k_ctx = max(k_ctx, 14)
        t_retrieve = time.perf_counter()
        context = retrieve_context(question, k=k_ctx)
        logger.info("/ask retrieve_context done in %.2fs", time.perf_counter() - t_retrieve)
        if general_intro:
            ctx0 = (context or "").strip()
            context = (
                f"{ACIBADEM_GENERAL_FOCUS_BLOCK}\n\n{ctx0}".strip()
                if ctx0
                else ACIBADEM_GENERAL_FOCUS_BLOCK.strip()
            )
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
        if cs_eng_q and not cs_course_catalog_q:
            ce_block = _ce_overview_context_block()
            ctx_body = (context or "").strip()
            context = f"{ce_block}\n\n{ctx_body}".strip() if ctx_body else ce_block
        # Varsayılan KAPALI: CPU'da tüm embedding matrisi + Ollama 4–7 dk'ı aşabiliyor.
        # Açmak için: ACU_COURSE_CATALOG_EMBED_AUGMENT=1 (veya true) — önbellek ısındıktan sonra daha hızlı.
        if cs_course_catalog_q and (
            (os.environ.get("ACU_COURSE_CATALOG_EMBED_AUGMENT") or "0").strip().lower()
            not in ("0", "false", "no")
        ):
            try:
                from chatbot.models import ChunkEmbedding as _CEMod

                emb_k = min(8, max(6, k_ctx // 2 + 2))
                obs_emb_n = _CEMod.objects.filter(chunk__source_type="obs").count()
                # Tek tarama: OBS embedding varsa sadece obs; yoksa tümü (çift tarama kaldırıldı).
                emb_chunks = _retrieve_top_chunks_by_embedding(
                    question,
                    k=emb_k,
                    source_type="obs" if obs_emb_n > 0 else None,
                )
                if emb_chunks:
                    lines: list[str] = []
                    for c in emb_chunks:
                        meta = " | ".join([p for p in [c.get("title") or "", c.get("url") or ""] if p])
                        t = (c.get("text") or "").strip()
                        if not t:
                            continue
                        lines.append(f"[{meta}]\n{t}" if meta else t)
                    if lines:
                        inject = "\n\n---\n\n".join(lines)
                        base = (context or "").strip()
                        context = (
                            f"{base}\n\n---\n\n"
                            f"[İlgili parçalar — anlamsal (embedding) arama]\n\n{inject}".strip()
                            if base
                            else f"[İlgili parçalar — anlamsal (embedding) arama]\n\n{inject}".strip()
                        )
            except Exception:
                logger.exception("course_catalog_embedding_augment_failed")
        selected_context, selected_sources, retrieved_chunks = _select_context_for_llm(
            question,
            context,
            max_chunks=int(os.environ.get("DJANGO_SELECTED_MAX_CHUNKS", "4")),
            max_chars=int(os.environ.get("DJANGO_SELECTED_MAX_CHARS", "4200")),
        )
        context = selected_context
        logger.info("SELECTED_CONTEXT_FILES=%s", selected_sources)
        logger.info(
            "OLLAMA_PRECHECK question=%r retrieved_chunks=%s selected_sources=%s context_chars=%s",
            question,
            retrieved_chunks,
            selected_sources,
            len(context),
        )
        if not context:
            logger.info("ANSWER_SOURCE=FALLBACK")
            logger.info("EXTRACTIVE_REASON=context_weak_or_unrelated")
            return _persist_assistant_reply(
                conv,
                _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                attach_followup=False,
                is_tr=is_tr,
                question=question,
            )
        if _is_extractive_question(question):
            logger.info("EXTRACTIVE_ATTEMPTED")
            extractive = _try_extractive_answer(question, context)
            if extractive:
                answer_text, reason = extractive
                logger.info("ANSWER_SOURCE=EXTRACTIVE")
                logger.info("EXTRACTIVE_FOUND")
                logger.info("EXTRACTIVE_REASON=%s", reason)
                return _persist_assistant_reply(
                    conv,
                    answer_text,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
                )
            logger.info("EXTRACTIVE_NOT_FOUND_CONTINUE_TO_LLM")

        max_context_chars = int(os.environ.get("DJANGO_MAX_CONTEXT_CHARS", "4200"))
        embed_augment_on = (
            (os.environ.get("ACU_COURSE_CATALOG_EMBED_AUGMENT") or "0").strip().lower()
            not in ("0", "false", "no")
        )
        if dept_cat:
            max_context_chars = max(
                max_context_chars,
                int(os.environ.get("DJANGO_DEPT_CATALOG_CONTEXT_CHARS", "10000")),
            )
        if cs_course_catalog_q and embed_augment_on:
            max_context_chars = max(
                max_context_chars,
                int(os.environ.get("DJANGO_COURSE_CATALOG_CONTEXT_CHARS", "14000")),
            )
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

        cs_eng_rules = ""
        if _cs_engineering_lisans_intent(question):
            if _cs_engineering_course_catalog_intent(question):
                cs_eng_rules = """
COMPUTER ENGINEERING — COURSE LIST / CURRICULUM QUESTION:
- The user asked for **concrete courses, codes, credits, semesters, or year-level curriculum** for Bilgisayar Mühendisliği (lisans).
- Answer **only** with information explicitly present in the Bağlam (retrieved chunks, e.g. OBS or official pages). List course names/codes/credits as they appear; you may group by semester/year **only if** the text supports it.
- Do **not** answer with a generic encyclopedic description of computer engineering or "typical" university subjects not named in the Bağlam.
- **Bilgisayar Programcılığı** (önlisans) is a different program: if the Bağlam is clearly about that program, say so briefly and do not present it as the engineering degree curriculum.
- If the Bağlam does not contain the requested year/course list, say clearly (in the user's language) that this list was not found in the retrieved documents — do not invent course names or codes.
"""
            else:
                cs_eng_rules = """
COMPUTER ENGINEERING vs PROGRAMMING:
- The context may begin with a **general overview** of Bilgisayar Mühendisliği (lisans). Use it only to explain the field, typical course areas, and labs/projects at a high level when the user did not ask for a specific course catalogue. State clearly that it is not the official course catalogue.
- **Bilgisayar Programcılığı** (önlisans) is a different program: do not describe its lab pages or curriculum as if they were the engineering degree. If lower context is only associate programming, mention the distinction in one short sentence and base the engineering explanation on the overview block.
- Do not invent specific course codes, credit counts, or prerequisite chains not stated in the context.
"""

        green_campus_rules = ""
        if campus_green_q:
            green_campus_rules = """
SUSTAINABLE / GREEN CAMPUS (sürdürülebilir kampüs):
- The user asks what a sustainable campus means or how the university approaches sustainability.
- If the Bağlam contains words like "sürdürülebilir", "sustainable", "çevre", "iklim", "karbon", "LEED", "yeşil", or similar, you MUST base your answer on those lines (paraphrase clearly). Short marketing lines are enough to give a useful explanation.
- Do NOT reply with only the stock phrase "Bu konuda elimde net bir bilgi bulunamadı" / "I couldn't find clear information about this" when any such wording appears in the Bağlam.
"""

        dept_catalog_rules = ""
        if dept_cat:
            dept_catalog_rules = """
FACULTY / DEPARTMENT OVERVIEW:
- The user wants faculties/schools/departments. Use **all** distinct names that appear across the Bağlam (scan every chunk).
- You may answer at length if needed to list everything found. Prefer bullets or clear grouping.
- If the Bağlam is incomplete vs the real university, say briefly that the list is only what appears in the retrieved text.
"""

        general_intro_rules = ""
        if general_intro:
            general_intro_rules = """
GENERAL UNIVERSITY INTRO:
- The user asked for a **broad** overview of Acıbadem University. Lead with its identity as a foundation university with major strengths in **health sciences, medicine, nursing, pharmacy/dentistry**, and links to healthcare/clinical training when the Bağlam supports this.
- Do **not** center the answer on Computer Engineering, data science, AI, or one department head unless the user explicitly asked about that program.
- Engineering and other faculties may appear as part of a balanced picture, not as the main headline.
"""

        prompt = f"""
You are an Acibadem University RAG assistant.

CORE RULES:
- Use ONLY information in CONTEXT.
- Do not use outside knowledge.
- Do not guess or invent facts.

LANGUAGE:
- The question language is {answer_language_instruction}.
- Final answer MUST be in {answer_language_instruction}.
- If context is in another language, translate only supported facts.

OUTPUT FORMAT:
- Keep the answer short, clear, and factual.
- Use bullet points for lists (departments, requirements, contacts, dates).
- For "which departments" questions, output only the department list as bullets.
- Do not add generic ending lines.

SOURCE LOYALTY:
- Use only consistent facts from context.
- If context snippets conflict, state that there is a conflict and advise checking the official website.
- Never invent person names, titles, URLs, course codes, fees, or dates.

FALLBACK RULE:
- If context does not clearly contain the answer, output EXACTLY one of:
  - Turkish: "Bu bilgi yerel veri kaynaklarında net olarak bulunamadı. En doğru ve güncel bilgi için Acıbadem Üniversitesi’nin resmi web sitesini kontrol etmeniz önerilir."
  - English: "This information was not clearly found in the local data sources. For the most accurate and up-to-date information, please check Acıbadem University’s official website."

{green_campus_rules}
{dept_catalog_rules}
{general_intro_rules}
{cs_eng_rules}
{address_rules}

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (context-only):
"""
        logger.info(
            "OLLAMA_INPUT question=%r retrieved_chunks=%s selected_sources=%s context_chars=%s prompt_chars=%s",
            question,
            retrieved_chunks,
            selected_sources,
            len(context),
            len(prompt),
        )
        t_llm = time.perf_counter()
        answer = ask_gemma(prompt)
        logger.info("/ask ask_gemma (primary) done in %.2fs", time.perf_counter() - t_llm)
        if answer == "__OLLAMA_TIMEOUT__":
            logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(prompt))
            logger.info("ANSWER_SOURCE=FALLBACK")
            return _persist_assistant_reply(
                conv,
                _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                attach_followup=False,
                is_tr=is_tr,
                question=question,
            )
        if (
            not (answer or "").strip()
            and len((context or "").strip()) > 180
            and retrieved_chunks >= 2
        ):
            refill = (
                "Aşağıdaki Bağlamı kullanarak soruyu Türkçe yanıtla. Boş bırakma; en az 2 anlamlı cümle yaz.\n\n"
                f"Bağlam:\n{context[:4000]}\n\nSoru:\n{question}"
                if is_tr
                else (
                    "Answer the question in English using the context below. Do not leave the answer empty; "
                    "at least 2 meaningful sentences.\n\n"
                    f"Context:\n{context[:4000]}\n\nQuestion:\n{question}"
                )
            )
            t_ref = time.perf_counter()
            answer = (ask_gemma(refill) or "").strip()
            logger.info("/ask ask_gemma empty-refill done in %.2fs", time.perf_counter() - t_ref)
            if answer == "__OLLAMA_TIMEOUT__":
                logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(refill))
                logger.info("ANSWER_SOURCE=FALLBACK")
                return _persist_assistant_reply(
                    conv,
                    _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
                )
        ctx_lc = (context or "").lower()
        if campus_green_q and ctx_lc and any(
            w in ctx_lc
            for w in (
                "sürdürülebilir",
                "surdurulebilir",
                "sustainable",
                "iklim",
                "çevre",
                "cevre",
                "yeşil",
                "yesil",
                "karbon",
                "leed",
                "eko",
            )
        ):
            if _answer_is_stock_no_info(answer):
                retry = (
                    "Aşağıdaki bağlamdan YALNIZCA yazılanları kullanarak soruyu Türkçe, **3–6 kısa cümle** ile yanıtla. "
                    "Uydurma bilgi ekleme. Bağlamda sürdürülebilirlik, çevre veya kampüsle ilgili ne varsa açıkla.\n\n"
                    f"Bağlam:\n{context[:4000]}\n\nSoru: {question}"
                    if is_tr
                    else (
                        "Answer in English using ONLY the context below (**3–6 short sentences**). "
                        "Do not invent facts. Explain any sustainability, environment, or campus-related wording.\n\n"
                        f"Context:\n{context[:4000]}\n\nQuestion: {question}"
                    )
                )
                retry_answer = (ask_gemma(retry) or "").strip()
                if retry_answer == "__OLLAMA_TIMEOUT__":
                    logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(retry))
                    logger.info("ANSWER_SOURCE=FALLBACK")
                    return _persist_assistant_reply(
                        conv,
                        _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                        attach_followup=False,
                        is_tr=is_tr,
                        question=question,
                    )
                answer = retry_answer or answer
        generic_retry_on = (os.environ.get("DJANGO_ENABLE_GENERIC_RETRY") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if (
            generic_retry_on
            and (
            _context_likely_relevant(question, context)
            and _answer_is_stock_no_info(answer)
            and len((context or "").strip()) > 120
            )
        ):
            retry_gen = (
                "Kullanıcının sorusu ile aşağıdaki bağlam arasında anlamlı kelime örtüşmesi var. "
                "Bağlamdan YALNIZCA desteklenen bilgileri kullanarak Türkçe, **3–6 kısa cümle** yaz. "
                "Uydurma. Bağlam gerçekten cevap vermiyorsa tek cümlede 'Bu konuda elimde net bir bilgi bulunamadı.' de.\n\n"
                f"Bağlam:\n{context[:4200]}\n\nSoru: {question}"
                if is_tr
                else (
                    "There is lexical overlap between the question and the context below. "
                    "Write a helpful answer in English using ONLY supported facts (**3–6 short sentences**). "
                    "Do not invent. If the context truly does not support an answer, output only: "
                    "I couldn't find clear information about this.\n\n"
                    f"Context:\n{context[:4200]}\n\nQuestion: {question}"
                )
            )
            retry_gen_answer = (ask_gemma(retry_gen) or "").strip()
            if retry_gen_answer == "__OLLAMA_TIMEOUT__":
                logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(retry_gen))
                logger.info("ANSWER_SOURCE=FALLBACK")
                return _persist_assistant_reply(
                    conv,
                    _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
                )
            answer = retry_gen_answer or answer
        if answer.startswith("Gemma error:"):
            return _persist_assistant_reply(conv, answer, status=502, as_detail=True)
        if not (answer or "").strip():
            return _persist_assistant_reply(
                conv,
                no_info_msg,
                attach_followup=True,
                is_tr=is_tr,
                question=question,
            )
        # Dil düzeltmesi: ikinci bir tam Ollama çağrısı dakikalarca sürebilir. Türkçe soruda yalnızca
        # cevap belirgin İngilizceyse çevir (ASCII Türkçe yanıtı yanlışlıkla İngilizce sanma).
        if is_tr:
            if _looks_english(answer):
                t_tr = time.perf_counter()
                answer = _translate_answer(answer, "tr")
                logger.info("/ask translate->tr done in %.2fs", time.perf_counter() - t_tr)
        else:
            if _looks_turkish(answer) and not _looks_english(answer):
                t_tr = time.perf_counter()
                answer = _translate_answer(answer, "en")
                logger.info("/ask translate->en done in %.2fs", time.perf_counter() - t_tr)
        if address_intent:
            answer = _strip_urls_plain_text(answer)
        logger.info("ANSWER_SOURCE=RAG_LLM")
        return _persist_assistant_reply(
            conv,
            answer,
            attach_followup=True,
            is_tr=is_tr,
            question=question,
        )
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
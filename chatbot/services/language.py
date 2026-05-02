"""Language-aware utilities used by the ask orchestrator and intent detectors.

Pure helpers only: no LLM calls, no Django request/response coupling. The
LLM-dependent ``_translate_answer`` stays in the legacy module until the LLM
client is extracted (refactor phase F5).
"""
from __future__ import annotations

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


__all__ = [
    "_TR_STOPWORDS",
    "_ascii_fold_turkish",
    "_detect_language",
    "_looks_turkish",
    "_looks_english",
    "_extract_keywords",
]

"""Heuristic extractive answers used before falling back to the LLM.

When the user's intent is "give me the list / the address / the person", we try to
synthesize a short answer directly from already-retrieved context blocks. If none of
the strategies fit, the orchestrator continues with the LLM.

Allowed dependency direction:
    extractive → services.context_select, services.intents, services.language
"""
from __future__ import annotations

import re

from chatbot.services.context_select import _split_context_blocks
from chatbot.services.intents import _extract_faculty_phrase
from chatbot.services.language import _ascii_fold_turkish


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


__all__ = [
    "_extractive_department_list",
    "_extractive_person_or_title",
    "_extractive_contact_or_address",
    "_try_extractive_answer",
]

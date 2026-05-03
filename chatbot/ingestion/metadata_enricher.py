"""
Metadata enrichment for ingested pages and chunks.

Why this exists
---------------
The crawler and the OIBS Bologna scraper produce reasonably-cleaned text
(see :mod:`chatbot.ingestion.content_cleaner`), but the only structural
hint they hand back is ``source_url``. That is not enough for the RAG
retriever to answer questions like "Bilgisayar Mühendisliği'nin 3.
yarıyıl dersleri nelerdir?" — the retriever needs filterable fields:
faculty, department, level, content_type, course_code, semester, etc.

This module turns three input shapes into a uniform metadata dict:

  1. A bare URL                         -> :func:`enrich_from_url`
  2. A Bologna program / info page      -> :func:`enrich_bologna_program`,
                                           :func:`enrich_bologna_info_page`
  3. A Bologna course                   -> :func:`enrich_bologna_course`

Plus a merge helper for stitching the URL/e-mail artefacts that the
plain-text cleaner extracts back onto the metadata blob.

Stable invariants
-----------------
* ``content_type`` is ALWAYS a lowercase snake_case ASCII string. The
  set of valid values lives on :class:`ContentType`. A module-load-time
  check (:func:`_validate_content_types`) refuses to import the module
  if a value is added that violates the rule.
* The Bologna program registry (:data:`BOLOGNA_PROGRAM_REGISTRY`) is
  the single source of truth for ``cur_unit/cur_sunit`` -> human-
  readable program mapping. To add a new program: append one line.
* All metadata returned is JSON-serialisable (``dict[str, Any]`` with
  primitive values only). It will be written to ``ScrapedPage.metadata``
  / ``PageChunk.metadata`` JSON columns by Step 3.3 / 3.4 wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import parse_qs, urlparse


# ---------------------------------------------------------------------------
# Content-type vocabulary
# ---------------------------------------------------------------------------


class ContentType:
    """Authoritative list of ``content_type`` values used across the RAG.

    All values are lowercase snake_case so retrieval-time filters can do
    plain string comparison. New values MUST follow the same convention;
    :func:`_validate_content_types` enforces this on import.
    """

    # --- Generic web (acibadem.edu.tr) ---
    HOMEPAGE = "homepage"
    GENERAL = "general"
    CONTACT = "contact"
    ANNOUNCEMENT = "announcement"
    NEWS = "news"
    EVENT = "event"
    FACULTY_PAGE = "faculty_page"
    DEPARTMENT_PAGE = "department_page"
    PROGRAM_PAGE = "program_page"
    CAMPUS_LIFE = "campus_life"
    ADMISSION = "admission"
    SCHOLARSHIP = "scholarship"
    LIBRARY = "library"
    INTERNATIONAL = "international"
    RESEARCH = "research"
    ABOUT_UNIVERSITY = "about_university"

    # --- OIBS Bologna program-level pages ---
    BOLOGNA_PROGRAM = "bologna_program"
    BOLOGNA_GOALS = "bologna_goals"
    BOLOGNA_ABOUT = "bologna_about"
    BOLOGNA_PROFILE = "bologna_profile"
    BOLOGNA_OFFICIALS = "bologna_officials"
    BOLOGNA_DEGREE = "bologna_degree"
    BOLOGNA_ADMISSION = "bologna_admission"
    BOLOGNA_FURTHER_STUDIES = "bologna_further_studies"
    BOLOGNA_GRADUATION = "bologna_graduation"
    BOLOGNA_PRIOR_LEARNING = "bologna_prior_learning"
    BOLOGNA_QUALIFICATION_RULES = "bologna_qualification_rules"
    BOLOGNA_OCCUPATION = "bologna_occupation"
    BOLOGNA_ACADEMIC_STAFF = "bologna_academic_staff"
    BOLOGNA_CONTACT = "bologna_contact"
    BOLOGNA_OUTCOMES = "bologna_outcomes"

    # --- OIBS Bologna course-level page ---
    BOLOGNA_COURSE = "bologna_course"


def _validate_content_types() -> None:
    """Refuse to load the module if a non-snake_case value sneaks in.

    Catches typos like ``"Bologna_Course"`` or ``"bologna course"`` at
    import time so a silent retrieval bug never reaches production.
    """
    seen: set[str] = set()
    for name in vars(ContentType):
        if name.startswith("_"):
            continue
        value = getattr(ContentType, name)
        if not isinstance(value, str):
            continue
        if value != value.lower():
            raise ValueError(f"ContentType.{name} must be lowercase: {value!r}")
        if " " in value or "-" in value:
            raise ValueError(
                f"ContentType.{name} must be snake_case (no spaces or dashes): {value!r}"
            )
        if not value.replace("_", "").isalnum():
            raise ValueError(
                f"ContentType.{name} must be ASCII alnum/underscore only: {value!r}"
            )
        if value in seen:
            raise ValueError(f"ContentType.{name} duplicates an existing value: {value!r}")
        seen.add(value)


_validate_content_types()


# ---------------------------------------------------------------------------
# Bologna program registry (extend as we widen the pilot)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BolognaProgramEntry:
    """One row in the OIBS program registry.

    ``cur_unit`` and ``cur_sunit`` are the OIBS query parameters that
    uniquely identify a program; everything else is the human-readable
    label we want to surface to the chatbot user.
    """

    cur_unit: int
    cur_sunit: int
    faculty: str
    department: str
    level: str  # "lisans" / "lisansüstü" / "önlisans"
    language: str  # ISO 639-1 ("tr", "en") or "tr+en" for bilingual
    program_slug: str  # stable, URL-safe slug for our own indexing


# Keyed by (cur_unit, cur_sunit) — the natural compound key in OIBS.
# Pilot scope (per project lead): only Bilgisayar Mühendisliği. Adding
# a new program is a one-line append.
BOLOGNA_PROGRAM_REGISTRY: dict[tuple[int, int], BolognaProgramEntry] = {
    (14, 6246): BolognaProgramEntry(
        cur_unit=14,
        cur_sunit=6246,
        faculty="Mühendislik ve Doğa Bilimleri Fakültesi",
        department="Bilgisayar Mühendisliği",
        level="lisans",
        language="en",
        program_slug="bilgisayar-muhendisligi-en",
    ),
    # Future programs go here, e.g.:
    # (14, 6247): BolognaProgramEntry(
    #     cur_unit=14, cur_sunit=6247,
    #     faculty="Mühendislik ve Doğa Bilimleri Fakültesi",
    #     department="Biyomedikal Mühendisliği", level="lisans",
    #     language="tr", program_slug="biyomedikal-muhendisligi",
    # ),
}


def lookup_bologna_program(cur_unit: int, cur_sunit: int) -> BolognaProgramEntry | None:
    """Return the registry entry for a program, or ``None`` if unknown.

    A miss is not an error — it just means the caller has to fall back
    to whatever it already knows about the program (e.g. the faculty /
    department names scraped from the page itself).
    """
    return BOLOGNA_PROGRAM_REGISTRY.get((cur_unit, cur_sunit))


# ---------------------------------------------------------------------------
# Slug -> display name maps (acibadem.edu.tr URL paths)
# ---------------------------------------------------------------------------

# These maps cover the slugs we expect to see during the pilot. Anything
# not in the map falls through to ``_slug_to_title`` which produces a
# best-effort title-case rendering (used as observability — actual
# faculty / department names should be added explicitly).

LEVEL_SLUG_MAP: dict[str, str] = {
    "lisans": "lisans",
    "lisansustu": "lisansüstü",
    "lisans-ustu": "lisansüstü",
    "onlisans": "önlisans",
    "on-lisans": "önlisans",
}

FACULTY_SLUG_MAP: dict[str, str] = {
    "tip-fakultesi": "Tıp Fakültesi",
    "dis-hekimligi-fakultesi": "Diş Hekimliği Fakültesi",
    "eczacilik-fakultesi": "Eczacılık Fakültesi",
    "saglik-bilimleri-fakultesi": "Sağlık Bilimleri Fakültesi",
    # The school renamed the engineering faculty in 2024; keep both
    # slugs alive so older URLs still resolve.
    "muhendislik-fakultesi": "Mühendislik ve Doğa Bilimleri Fakültesi",
    "muhendislik-ve-doga-bilimleri-fakultesi": "Mühendislik ve Doğa Bilimleri Fakültesi",
    "fen-edebiyat-fakultesi": "Fen-Edebiyat Fakültesi",
    "saglik-hizmetleri-meslek-yuksekokulu": "Sağlık Hizmetleri Meslek Yüksekokulu",
    "saglik-bilimleri-enstitusu": "Sağlık Bilimleri Enstitüsü",
}

DEPARTMENT_SLUG_MAP: dict[str, str] = {
    "bilgisayar-muhendisligi": "Bilgisayar Mühendisliği",
    "biyomedikal-muhendisligi": "Biyomedikal Mühendisliği",
    "endustri-muhendisligi": "Endüstri Mühendisliği",
    "elektrik-elektronik-muhendisligi": "Elektrik-Elektronik Mühendisliği",
    "molekuler-biyoloji-ve-genetik": "Moleküler Biyoloji ve Genetik",
    "psikoloji": "Psikoloji",
    "hemsirelik": "Hemşirelik",
    "fizyoterapi-ve-rehabilitasyon": "Fizyoterapi ve Rehabilitasyon",
    "beslenme-ve-diyetetik": "Beslenme ve Diyetetik",
    "saglik-yonetimi": "Sağlık Yönetimi",
    "tip": "Tıp",
    "dis-hekimligi": "Diş Hekimliği",
    "eczacilik": "Eczacılık",
}

# Top-level path segments on www.acibadem.edu.tr that map to a stable
# content_type without further inspection.
PATH_TO_CONTENT_TYPE: dict[str, str] = {
    "iletisim": ContentType.CONTACT,
    "duyurular": ContentType.ANNOUNCEMENT,
    "duyuru": ContentType.ANNOUNCEMENT,
    "haberler": ContentType.NEWS,
    "haber": ContentType.NEWS,
    "etkinlikler": ContentType.EVENT,
    "etkinlik": ContentType.EVENT,
    "kampus-hayati": ContentType.CAMPUS_LIFE,
    "ogrenci-yasami": ContentType.CAMPUS_LIFE,
    "aday-ogrenci": ContentType.ADMISSION,
    "kayit-kabul": ContentType.ADMISSION,
    "burs": ContentType.SCHOLARSHIP,
    "burslar": ContentType.SCHOLARSHIP,
    "kutuphane": ContentType.LIBRARY,
    "uluslararasi": ContentType.INTERNATIONAL,
    "uluslararasi-iliskiler": ContentType.INTERNATIONAL,
    "arastirma": ContentType.RESEARCH,
    "ar-ge": ContentType.RESEARCH,
    "hakkimizda": ContentType.ABOUT_UNIVERSITY,
    "kurumsal": ContentType.ABOUT_UNIVERSITY,
}


# ---------------------------------------------------------------------------
# Bologna-specific path / key maps
# ---------------------------------------------------------------------------

# OIBS aspx filenames -> our content_type. Keys are lower-cased to make
# the lookup case-insensitive (OIBS is inconsistent: progAbout.aspx and
# progAccessFurhterStudies.aspx happily coexist).
BOLOGNA_ASPX_TO_CONTENT_TYPE: dict[str, str] = {
    "proggoalsobjectives.aspx": ContentType.BOLOGNA_GOALS,
    "progabout.aspx": ContentType.BOLOGNA_ABOUT,
    "progprofile.aspx": ContentType.BOLOGNA_PROFILE,
    "progofficials.aspx": ContentType.BOLOGNA_OFFICIALS,
    "progdegree.aspx": ContentType.BOLOGNA_DEGREE,
    "progadmissionreq.aspx": ContentType.BOLOGNA_ADMISSION,
    "progaccessfurhterstudies.aspx": ContentType.BOLOGNA_FURTHER_STUDIES,
    "proggraduationreq.aspx": ContentType.BOLOGNA_GRADUATION,
    "progrecogpriorlearning.aspx": ContentType.BOLOGNA_PRIOR_LEARNING,
    "progqualifyreqreg.aspx": ContentType.BOLOGNA_QUALIFICATION_RULES,
    "progoccupationalprof.aspx": ContentType.BOLOGNA_OCCUPATION,
    "progacademicstaff.aspx": ContentType.BOLOGNA_ACADEMIC_STAFF,
    "progcontact.aspx": ContentType.BOLOGNA_CONTACT,
    "proglearnoutcomes.aspx": ContentType.BOLOGNA_OUTCOMES,
    "progcourses.aspx": ContentType.BOLOGNA_PROGRAM,
    "index.aspx": ContentType.BOLOGNA_PROGRAM,
}

# The same mapping by ``info_pages`` dictionary key (the stable string
# the Bologna scraper writes alongside each page it captures).
BOLOGNA_INFO_KEY_TO_CONTENT_TYPE: dict[str, str] = {
    "goals_objectives": ContentType.BOLOGNA_GOALS,
    "about": ContentType.BOLOGNA_ABOUT,
    "profile": ContentType.BOLOGNA_PROFILE,
    "officials": ContentType.BOLOGNA_OFFICIALS,
    "degree": ContentType.BOLOGNA_DEGREE,
    "admission": ContentType.BOLOGNA_ADMISSION,
    "further_studies": ContentType.BOLOGNA_FURTHER_STUDIES,
    "graduation": ContentType.BOLOGNA_GRADUATION,
    "prior_learning": ContentType.BOLOGNA_PRIOR_LEARNING,
    "qualification_rules": ContentType.BOLOGNA_QUALIFICATION_RULES,
    "occupation": ContentType.BOLOGNA_OCCUPATION,
    "academic_staff": ContentType.BOLOGNA_ACADEMIC_STAFF,
    "contact": ContentType.BOLOGNA_CONTACT,
}


# ---------------------------------------------------------------------------
# Slug helper
# ---------------------------------------------------------------------------


def _slug_to_title(slug: str) -> str:
    """Best-effort fallback when a slug is not in our explicit maps.

    Used only for observability — code paths that depend on accurate
    Turkish names should consult the explicit slug maps. Bare slugs are
    still better than nothing so the operator can spot an unmapped
    department in the metadata and add it to the dictionary.
    """
    if not slug:
        return ""
    return slug.replace("-", " ").replace("_", " ").strip().title()


# ---------------------------------------------------------------------------
# URL-based enricher
# ---------------------------------------------------------------------------


def enrich_from_url(url: str) -> dict[str, Any]:
    """Return a metadata dict derived purely from a URL.

    Detects the host and dispatches:
      * ``obs.acibadem.edu.tr/oibs/bologna/...``   -> Bologna pages.
      * ``www.acibadem.edu.tr/akademik/.../...``   -> faculty/department.
      * Top-level paths like ``/iletisim``, ``/duyurular``.

    Unknown URLs return ``{"source_url": url, "content_type": "general"}``
    so callers can still write a chunk without crashing.
    """
    if not url:
        return {"content_type": ContentType.GENERAL}

    parsed = urlparse(url)
    host = (parsed.netloc or "").lower()
    path = (parsed.path or "").strip("/").lower()
    parts = [p for p in path.split("/") if p]
    query = parse_qs(parsed.query or "")

    metadata: dict[str, Any] = {"source_url": url}
    if host:
        metadata["host"] = host

    if host.startswith("obs."):
        return _enrich_bologna_url(metadata, parts, query)

    return _enrich_www_url(metadata, parts)


def _enrich_bologna_url(
    metadata: dict[str, Any],
    parts: list[str],
    query: dict[str, list[str]],
) -> dict[str, Any]:
    """Subroutine of :func:`enrich_from_url` for ``obs.acibadem.edu.tr``."""
    cur_sunit = _first_int(query.get("curSunit"))
    cur_unit = _first_int(query.get("curUnit"))
    if cur_sunit is not None:
        metadata["cur_sunit"] = cur_sunit
    if cur_unit is not None:
        metadata["cur_unit"] = cur_unit

    if cur_sunit is not None:
        # Try every known cur_unit if the URL didn't carry one.
        for entry in BOLOGNA_PROGRAM_REGISTRY.values():
            if entry.cur_sunit == cur_sunit and (
                cur_unit is None or entry.cur_unit == cur_unit
            ):
                metadata["faculty"] = entry.faculty
                metadata["department"] = entry.department
                metadata["level"] = entry.level
                metadata["language"] = entry.language
                metadata["program_slug"] = entry.program_slug
                metadata.setdefault("cur_unit", entry.cur_unit)
                break

    if parts:
        last = parts[-1].lower()
        metadata["content_type"] = BOLOGNA_ASPX_TO_CONTENT_TYPE.get(
            last, ContentType.BOLOGNA_PROGRAM
        )
    else:
        metadata["content_type"] = ContentType.BOLOGNA_PROGRAM
    return metadata


def _enrich_www_url(metadata: dict[str, Any], parts: list[str]) -> dict[str, Any]:
    """Subroutine of :func:`enrich_from_url` for ``www.acibadem.edu.tr``."""
    if not parts:
        metadata["content_type"] = ContentType.HOMEPAGE
        return metadata

    first = parts[0]

    # Top-level shortcut paths (iletisim, duyurular, ...).
    direct = PATH_TO_CONTENT_TYPE.get(first)
    if direct is not None:
        metadata["content_type"] = direct
        return metadata

    if first == "akademik" and len(parts) >= 2:
        return _enrich_academic_path(metadata, parts)

    metadata["content_type"] = ContentType.GENERAL
    return metadata


def _enrich_academic_path(
    metadata: dict[str, Any], parts: list[str]
) -> dict[str, Any]:
    """Handle ``/akademik/<level>/<faculty>/<bolumler>/<department>``.

    Each segment refines the metadata; missing segments simply leave
    that field unset rather than producing a partial-match guess.
    """
    metadata["content_type"] = ContentType.GENERAL  # tightened below

    level_slug = parts[1] if len(parts) >= 2 else ""
    level = LEVEL_SLUG_MAP.get(level_slug)
    if level:
        metadata["level"] = level

    if len(parts) >= 3:
        faculty_slug = parts[2]
        metadata["faculty"] = FACULTY_SLUG_MAP.get(faculty_slug) or _slug_to_title(
            faculty_slug
        )
        metadata["faculty_slug"] = faculty_slug
        metadata["content_type"] = ContentType.FACULTY_PAGE

    # Acibadem path convention: /akademik/<level>/<faculty>/bolumler/<dept>
    if len(parts) >= 5 and parts[3] == "bolumler":
        dept_slug = parts[4]
        metadata["department"] = DEPARTMENT_SLUG_MAP.get(dept_slug) or _slug_to_title(
            dept_slug
        )
        metadata["department_slug"] = dept_slug
        metadata["content_type"] = ContentType.DEPARTMENT_PAGE

    return metadata


# ---------------------------------------------------------------------------
# Bologna-aware enrichers (used by the Bologna ingest command)
# ---------------------------------------------------------------------------


def enrich_bologna_program(
    *,
    faculty: str,
    department: str,
    cur_unit: int,
    cur_sunit: int,
    program_url: str | None = None,
    level: str = "lisans",
    language: str = "tr",
) -> dict[str, Any]:
    """Metadata blob for the program-overview ScrapedPage.

    Used once per program — the entry point that aggregates program
    outcomes, Markdown overview, and links to the per-page chunks.
    """
    metadata: dict[str, Any] = {
        "content_type": ContentType.BOLOGNA_PROGRAM,
        "faculty": faculty,
        "department": department,
        "level": level,
        "language": language,
        "cur_unit": cur_unit,
        "cur_sunit": cur_sunit,
    }
    if program_url:
        metadata["source_url"] = program_url

    entry = lookup_bologna_program(cur_unit, cur_sunit)
    if entry is not None:
        metadata["program_slug"] = entry.program_slug
        # Registry takes precedence over caller-supplied values when set,
        # so a typo on the caller side cannot pollute the index.
        metadata["faculty"] = entry.faculty
        metadata["department"] = entry.department
        metadata["level"] = entry.level
        metadata["language"] = entry.language
    return metadata


def enrich_bologna_info_page(
    info_key: str,
    *,
    faculty: str,
    department: str,
    cur_unit: int,
    cur_sunit: int,
    page_url: str | None = None,
    level: str = "lisans",
    language: str = "tr",
) -> dict[str, Any]:
    """Metadata for one page captured under ``program.info_pages``.

    ``info_key`` is the stable identifier emitted by the Bologna
    scraper (``"about"``, ``"contact"``, ``"goals_objectives"`` ...).
    """
    metadata = enrich_bologna_program(
        faculty=faculty,
        department=department,
        cur_unit=cur_unit,
        cur_sunit=cur_sunit,
        program_url=page_url,
        level=level,
        language=language,
    )
    metadata["content_type"] = BOLOGNA_INFO_KEY_TO_CONTENT_TYPE.get(
        info_key, ContentType.BOLOGNA_PROGRAM
    )
    metadata["info_page_key"] = info_key
    return metadata


def enrich_bologna_course(
    *,
    code: str,
    name: str,
    faculty: str,
    department: str,
    cur_unit: int,
    cur_sunit: int,
    semester: int | None = None,
    course_type: str | None = None,
    ects: float | None = None,
    credit_theory: float | None = None,
    credit_practice: float | None = None,
    credit_lab: float | None = None,
    credit_total: float | None = None,
    delivery_mode: str | None = None,
    detail_url: str | None = None,
    postback_target: str | None = None,
    level: str = "lisans",
    language: str = "tr",
) -> dict[str, Any]:
    """Metadata for one Bologna course ScrapedPage.

    Bare numeric / string course attributes are passed through as-is;
    ``content_type`` and the program context are filled from the
    registry where possible.
    """
    metadata: dict[str, Any] = {
        "content_type": ContentType.BOLOGNA_COURSE,
        "faculty": faculty,
        "department": department,
        "level": level,
        "language": language,
        "cur_unit": cur_unit,
        "cur_sunit": cur_sunit,
        "course_code": code,
        "course_name": name,
    }
    if semester is not None:
        metadata["semester"] = int(semester)
    if course_type:
        metadata["course_type"] = course_type
    if ects is not None:
        metadata["ects"] = float(ects)
    if credit_theory is not None:
        metadata["credit_theory"] = float(credit_theory)
    if credit_practice is not None:
        metadata["credit_practice"] = float(credit_practice)
    if credit_lab is not None:
        metadata["credit_lab"] = float(credit_lab)
    if credit_total is not None:
        metadata["credit_total"] = float(credit_total)
    if delivery_mode:
        metadata["delivery_mode"] = delivery_mode
    if detail_url and not detail_url.startswith("javascript:"):
        metadata["source_url"] = detail_url
    if postback_target:
        # Stored opaquely so a future drill-down step can fire the
        # postback without re-scraping the curriculum view.
        metadata["postback_target"] = postback_target

    entry = lookup_bologna_program(cur_unit, cur_sunit)
    if entry is not None:
        metadata["program_slug"] = entry.program_slug
        metadata["faculty"] = entry.faculty
        metadata["department"] = entry.department
        metadata["level"] = entry.level
        metadata["language"] = entry.language
    return metadata


# ---------------------------------------------------------------------------
# Merge URL/e-mail artefacts produced by the plain-text cleaner
# ---------------------------------------------------------------------------


def merge_extracted_artifacts(
    metadata: dict[str, Any],
    *,
    urls: Iterable[str] | None = None,
    emails: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fold URL / e-mail lists from :class:`CleanedTextResult` into
    the metadata blob without overwriting an existing ``source_url``.

    Decision (project lead, Step 3.1):
      * ``source_url`` stays as the canonical page URL (set by the
        caller / the URL-based enricher); other URLs landed in the
        body text become ``related_urls``.
      * ``contact_emails`` collects e-mail addresses found in the
        body so contact-page chunks keep their answer payload while
        the LLM no longer has to read the literal address.
    """
    out: dict[str, Any] = dict(metadata)
    url_list = list(_dedup_keep_order(urls or ()))
    if url_list:
        canonical = out.get("source_url")
        if canonical:
            related = [u for u in url_list if u != canonical]
        else:
            out["source_url"] = url_list[0]
            related = url_list[1:]
        if related:
            existing = out.get("related_urls") or []
            out["related_urls"] = list(_dedup_keep_order([*existing, *related]))

    email_list = list(_dedup_keep_order((e.lower() for e in (emails or ()))))
    if email_list:
        existing = out.get("contact_emails") or []
        out["contact_emails"] = list(
            _dedup_keep_order([*existing, *(e.lower() for e in email_list)])
        )
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _first_int(values: list[str] | None) -> int | None:
    if not values:
        return None
    raw = values[0]
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _dedup_keep_order(values: Iterable[str]) -> Iterable[str]:
    seen: set[str] = set()
    for v in values:
        if not v or v in seen:
            continue
        seen.add(v)
        yield v


__all__ = (
    "ContentType",
    "BolognaProgramEntry",
    "BOLOGNA_PROGRAM_REGISTRY",
    "BOLOGNA_ASPX_TO_CONTENT_TYPE",
    "BOLOGNA_INFO_KEY_TO_CONTENT_TYPE",
    "FACULTY_SLUG_MAP",
    "DEPARTMENT_SLUG_MAP",
    "LEVEL_SLUG_MAP",
    "PATH_TO_CONTENT_TYPE",
    "lookup_bologna_program",
    "enrich_from_url",
    "enrich_bologna_program",
    "enrich_bologna_info_page",
    "enrich_bologna_course",
    "merge_extracted_artifacts",
)

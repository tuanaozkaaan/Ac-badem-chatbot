"""Heuristic-regex query parser for the RAG retriever.

Why this exists
---------------
Retrieval quality on a small (single-program) Bologna corpus collapses
when the embedding ranker has to compete with hundreds of phrases that
all mention "Bilgisayar Mühendisliği". A user asking
"Bilgisayar Mühendisliği bölüm başkanı kim?" really means *one* page
(``content_type='bologna_officials'`` AND ``department='Bilgisayar
Mühendisliği'``); answering with the program-overview page just because
its cosine score is marginally higher would be wrong even if the LLM
post-processes nicely.

Rather than burn another LLM round-trip on extraction (which would
double tail latency on a local Gemma 7B), we parse the query with a
small set of curated regex patterns. The parser produces a
:class:`QueryFilters` instance that the embedding retriever turns into
PostgreSQL ``JSONB`` ``WHERE`` clauses against ``PageChunk.metadata``.

Stable invariants
-----------------
* ``content_type`` values are taken straight from
  :class:`chatbot.ingestion.metadata_enricher.ContentType` so a typo on
  one side is impossible — module import will fail loudly.
* All canonical names (department / faculty) come from the same source
  of truth as :data:`metadata_enricher.DEPARTMENT_SLUG_MAP` /
  :data:`FACULTY_SLUG_MAP`. A regex pattern adding a new alias is the
  only place a contributor needs to touch.
* The function is pure: same string in → same QueryFilters out, no
  network, no DB, no caches. Safe to call on every request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from chatbot.ingestion.metadata_enricher import ContentType


@dataclass(frozen=True)
class QueryFilters:
    """Structured filters extracted from a free-text user question.

    Each attribute corresponds 1:1 to a key on ``PageChunk.metadata``,
    so the retrieval layer can map this object into a JSONB ``WHERE``
    clause without further translation. ``content_types`` is a *tuple*
    (not a set) so equality / hashing stays deterministic for tests.
    """

    faculty: str | None = None
    department: str | None = None
    course_code: str | None = None
    semester: int | None = None
    content_types: tuple[str, ...] = ()
    matched_terms: tuple[str, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not (
            self.faculty
            or self.department
            or self.course_code
            or self.semester
            or self.content_types
        )


# ---------------------------------------------------------------------------
# Turkish-aware boundary
# ---------------------------------------------------------------------------
# Python's ``\b`` is Unicode-aware: Turkish letters like ``ı`` count as
# word characters, so ``\bba[şs]kan\b`` will NOT match "başkanı" because
# there is no word boundary between "başkan" and "ı". Every regex below
# therefore allows an arbitrary trailing ``\w*`` after the stem so that
# Turkish inflectional suffixes (-ı, -in, -de, -lerine, ...) do not
# block a match. Doing this once at the end of each alternation is
# uglier than a helper, so we just standardise on the convention.
DEPARTMENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:"
            r"bilgisayar\s*m[üu]hendisli[gğ]i\w*"
            r"|bilgisayar\s+m[üu]h(?!\.\s*dr)\w*"
            r"|computer\s+(?:science|engineering)\w*"
            r")",
            re.IGNORECASE,
        ),
        "Bilgisayar Mühendisliği",
    ),
    (
        re.compile(
            r"\b(?:biyomedikal\s*m[üu]hendisli[gğ]i\w*|biomedical\s+engineering\w*)",
            re.IGNORECASE,
        ),
        "Biyomedikal Mühendisliği",
    ),
    (
        re.compile(
            r"\b(?:end[üu]stri\s*m[üu]hendisli[gğ]i\w*|industrial\s+engineering\w*)",
            re.IGNORECASE,
        ),
        "Endüstri Mühendisliği",
    ),
    (
        re.compile(
            r"\b(?:elektrik[\s\-]?elektronik(?:\s*m[üu]hendisli[gğ]i\w*)?"
            r"|electrical[\s\-]?electronics?\s+engineering\w*)",
            re.IGNORECASE,
        ),
        "Elektrik-Elektronik Mühendisliği",
    ),
    (
        re.compile(
            r"\b(?:molek[üu]ler\s+biyoloji(?:\s+ve\s+genetik)?\w*"
            r"|mbg\b|molecular\s+biology(?:\s+and\s+genetics)?\w*)",
            re.IGNORECASE,
        ),
        "Moleküler Biyoloji ve Genetik",
    ),
    (
        re.compile(r"\b(?:psikoloji\w*|psychology\w*)", re.IGNORECASE),
        "Psikoloji",
    ),
    (
        re.compile(r"\b(?:hem[şs]irelik\w*|nursing\w*)", re.IGNORECASE),
        "Hemşirelik",
    ),
    (
        re.compile(
            r"\b(?:fizyoterapi(?:\s+ve\s+rehabilitasyon)?\w*|physiotherapy\w*)",
            re.IGNORECASE,
        ),
        "Fizyoterapi ve Rehabilitasyon",
    ),
    (
        re.compile(
            r"\b(?:beslenme(?:\s+ve\s+diyetetik)?\w*"
            r"|nutrition(?:\s+and\s+dietetics)?\w*)",
            re.IGNORECASE,
        ),
        "Beslenme ve Diyetetik",
    ),
    (
        re.compile(
            r"\b(?:sa[gğ]l[ıi]k\s+y[öo]netimi\w*|health\s+management\w*)",
            re.IGNORECASE,
        ),
        "Sağlık Yönetimi",
    ),
    (
        re.compile(r"\b(?:di[şs]\s+hekimli[gğ]i\w*|dentistry\w*)", re.IGNORECASE),
        "Diş Hekimliği",
    ),
    (
        re.compile(r"\b(?:eczac[ıi]l[ıi]k\w*|pharmacy\w*)", re.IGNORECASE),
        "Eczacılık",
    ),
    # Generic "tıp" without "fakültesi" is intentionally not in this list
    # because it would also match "tıbbi" / "tıbbi terimler" off-topic.
)


# ---------------------------------------------------------------------------
# Faculty patterns
# ---------------------------------------------------------------------------
FACULTY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b(?:m[üu]hendislik(?:\s+ve\s+do[ğg]a\s+bilimleri)?\s+fak[üu]ltesi\w*"
            r"|engineering(?:\s+and\s+natural\s+sciences)?\s+faculty\w*"
            r"|faculty\s+of\s+engineering(?:\s+and\s+natural\s+sciences)?\w*)",
            re.IGNORECASE,
        ),
        "Mühendislik ve Doğa Bilimleri Fakültesi",
    ),
    (
        re.compile(
            r"\b(?:t[ıi]p\s+fak[üu]ltesi\w*"
            r"|school\s+of\s+medicine|medical\s+faculty\w*|faculty\s+of\s+medicine)",
            re.IGNORECASE,
        ),
        "Tıp Fakültesi",
    ),
    (
        re.compile(
            r"\b(?:di[şs]\s+hekimli[gğ]i\s+fak[üu]ltesi\w*"
            r"|school\s+of\s+dentistry|faculty\s+of\s+dentistry)",
            re.IGNORECASE,
        ),
        "Diş Hekimliği Fakültesi",
    ),
    (
        re.compile(
            r"\b(?:eczac[ıi]l[ıi]k\s+fak[üu]ltesi\w*"
            r"|school\s+of\s+pharmacy|faculty\s+of\s+pharmacy)",
            re.IGNORECASE,
        ),
        "Eczacılık Fakültesi",
    ),
    (
        re.compile(
            r"\b(?:sa[gğ]l[ıi]k\s+bilimleri\s+fak[üu]ltesi\w*"
            r"|health\s+sciences\s+faculty\w*|faculty\s+of\s+health\s+sciences)",
            re.IGNORECASE,
        ),
        "Sağlık Bilimleri Fakültesi",
    ),
)


# ---------------------------------------------------------------------------
# Department -> faculty inference (pilot scope)
# ---------------------------------------------------------------------------
# When the user names a department but not its faculty, we fill in the
# faculty so the metadata WHERE-clause can use both columns. Edit this
# map together with DEPARTMENT_PATTERNS when a new department joins the
# pilot. Conservative on purpose: a missing entry yields ``None`` and we
# simply skip the faculty filter (department alone is selective enough).
DEPARTMENT_TO_FACULTY: dict[str, str] = {
    "Bilgisayar Mühendisliği": "Mühendislik ve Doğa Bilimleri Fakültesi",
    "Biyomedikal Mühendisliği": "Mühendislik ve Doğa Bilimleri Fakültesi",
    "Endüstri Mühendisliği": "Mühendislik ve Doğa Bilimleri Fakültesi",
    "Elektrik-Elektronik Mühendisliği": "Mühendislik ve Doğa Bilimleri Fakültesi",
    "Moleküler Biyoloji ve Genetik": "Mühendislik ve Doğa Bilimleri Fakültesi",
    "Hemşirelik": "Sağlık Bilimleri Fakültesi",
    "Fizyoterapi ve Rehabilitasyon": "Sağlık Bilimleri Fakültesi",
    "Beslenme ve Diyetetik": "Sağlık Bilimleri Fakültesi",
    "Sağlık Yönetimi": "Sağlık Bilimleri Fakültesi",
    "Eczacılık": "Eczacılık Fakültesi",
    "Diş Hekimliği": "Diş Hekimliği Fakültesi",
}


# ---------------------------------------------------------------------------
# Course code
# ---------------------------------------------------------------------------
# Acıbadem OBS uses 2-4 letter prefixes ("CSE", "ENG", "MAT", "PHY",
# "CHE", ...) followed by a 3-digit number, optionally separated by a
# space, dash or underscore. We normalise to ``UPPERPREFIX###``.
COURSE_CODE_RE = re.compile(
    r"\b([A-Za-zÇĞİÖŞÜçğıöşü]{2,4})\s*[-_\s]?\s*(\d{3,4})\b"
)

# Stand-alone words that sometimes pass the course-code regex above but
# are clearly not codes ("DIS 101" → "DIS101" feels off; "TUR 101" →
# OK). Only filter the truly noisy ones.
_COURSE_CODE_BLOCKLIST: frozenset[str] = frozenset({
    # The course-code regex would match "no:101" → "NO101"; ASCII-fold
    # would also catch a Turkish phrase like "ne 101 bunlar"; keep these
    # tight and additive.
})


# ---------------------------------------------------------------------------
# Semester / yarıyıl
# ---------------------------------------------------------------------------
SEMESTER_RE = re.compile(
    r"\b(\d{1,2})\s*\.?\s*(?:yar[ıi]y[ıi]l\w*|d[öo]nem\w*|semester\w*|term\w*)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Intent → content_type
# ---------------------------------------------------------------------------
# Multiple intents can fire (e.g. "Bilgisayar Müh. iletişim ve bölüm
# başkanı" matches both ``bologna_contact`` and ``bologna_officials``);
# we keep them all and let retrieval decide. Order top-to-bottom only
# matters for ``matched_terms`` debug output.
INTENT_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (
        re.compile(
            r"\b(?:b[öo]l[üu]m\s+ba[şs]kan\w*|program\s+ba[şs]kan\w*"
            r"|department\s+(?:head|chair)\w*|chair\s+of\s+(?:the\s+)?department\w*"
            r"|m[üu]d[üu]r[üu]?(?!\s+yard[ıi]m)\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_OFFICIALS,),
    ),
    (
        re.compile(
            r"\b(?:akademik\s+personel\w*|hocalar\w*"
            r"|[öo][ğg]retim\s+[üu]yes[iı]\w*"
            r"|faculty\s+member\w*|academic\s+staff\w*|teaching\s+staff\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_ACADEMIC_STAFF,),
    ),
    (
        re.compile(
            r"\b(?:ileti[şs]im\w*|e[\s\-]?posta\w*|e[\s\-]?mail\w*"
            r"|telefon\s+(?:numaras|no)\w*"
            r"|contact(?:\s+info)?\w*|reach\s+out)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_CONTACT, ContentType.CONTACT),
    ),
    (
        re.compile(
            r"\b(?:[öo][ğg]renim\s+[çc][ıi]kt\w*|program\s+[çc][ıi]kt\w*"
            r"|learning\s+outcomes?\w*|program\s+outcomes?\w*|kazan[ıi]m\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_PROGRAM, ContentType.BOLOGNA_OUTCOMES),
    ),
    (
        re.compile(
            r"\b(?:mezuniyet\w*"
            r"|graduation\s+(?:requirement\w*|criteria\w*)"
            r"|mezun\s+olma\s+ko[şs]ul\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_GRADUATION,),
    ),
    (
        re.compile(
            r"\b(?:kabul\s+(?:[şs]art\w*|ko[şs]ul\w*|gereksinim\w*)"
            r"|admission\s+req\w*|nas[ıi]l\s+ba[şs]vur\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_ADMISSION,),
    ),
    (
        # Course / curriculum intent. Three-tier match so we cover both very
        # specific phrases ("müfredatı") AND the bare plurals that simply ask
        # for the program's course list ("dersler", "courses"). The Adım 5.0
        # cleanup removed a hard-coded CS catalog fallback; without these
        # plurals the parser tagged "Bilgisayar mühendisliği dersleri nedir?"
        # with no content_type, the hybrid retriever then returned program-
        # overview chunks, and the LLM bailed with "no info".
        re.compile(
            # 1) explicit catalog phrases
            r"\b(?:m[üu]fredat\w*|curriculum\w*"
            # 2) "ders" + qualifier (kodu, listesi, kataloğu, içeriği, planı, programı)
            r"|ders\s+(?:kodu\w*|listesi\w*|katalo[gğ]u\w*|i[çc]eri[gğ]i\w*"
            r"|plan[ıi]?\w*|program[ıi]?\w*)"
            # 3) bare plural Turkish ("dersler", "dersleri", "derslerini",
            #    "derslerine", "derslerin"). Plural-only on purpose: "ders"
            #    alone (singular) would over-match conversational uses like
            #    "bu ders güzeldi" that have nothing to do with catalogs.
            r"|dersler\w*"
            # 4) English equivalents — word-boundary so "coursework"/"coursing"
            #    do NOT match. The optional "list/catalog/plan" qualifier is
            #    accepted but not required.
            r"|courses?\b|course\s+(?:list\w*|catalog\w*|plan\w*)"
            # 5) ECTS / credit cues
            r"|akts\w*|ects\w*|kredi\w*"
            # 6) "güz/bahar yarıyılı" (specific term)
            r"|(?:g[üu]z|bahar)\s+yar[ıi]y[ıi]l\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_COURSE, ContentType.BOLOGNA_PROGRAM),
    ),
    (
        re.compile(
            r"\b(?:[üu]st\s+[öo][ğg]renim\w*|further\s+stud(?:y|ies)\w*"
            r"|y[üu]ksek\s+lisans\s+ge[çc]i[şs]\w*"
            r"|graduate\s+stud\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_FURTHER_STUDIES,),
    ),
    (
        re.compile(
            r"\b(?:mezun\w*\s+ne(?:reler)?\s+(?:i[şs]|[çc]al)\w*"
            r"|i[şs]\s+olanak\w*|kariyer\w*"
            r"|career\s+opportunit\w*|occupation\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_OCCUPATION,),
    ),
    (
        re.compile(
            r"\b(?:al[ıi]nacak\s+derece\w*|(?:bachelor|master|lisans)\s+derece\w*"
            r"|diploma\w*|degree\s+award\w*)",
            re.IGNORECASE,
        ),
        (ContentType.BOLOGNA_DEGREE,),
    ),
    (
        re.compile(
            r"\b(?:duyuru\w*|haber\w*|etkinlik\w*"
            r"|announcement\w*|news\b|event\w*)",
            re.IGNORECASE,
        ),
        (ContentType.ANNOUNCEMENT, ContentType.NEWS, ContentType.EVENT),
    ),
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def parse_query(question: str) -> QueryFilters:
    """Extract structured retrieval filters from a free-text question.

    Strategy
    --------
    1. Try each :data:`DEPARTMENT_PATTERNS` row; keep the FIRST match
       (patterns are ordered most-specific to least-specific).
    2. Try :data:`FACULTY_PATTERNS` independently. If we matched a
       department but no faculty, fill the faculty in via
       :data:`DEPARTMENT_TO_FACULTY` (best-effort; missing entry → ``None``).
    3. Pull the first :data:`COURSE_CODE_RE` hit and normalise it.
    4. Pull the first :data:`SEMESTER_RE` hit; clamp to 1..12 because
       a "yarıyıl" can never be 0 or 30 in a real curriculum.
    5. Walk :data:`INTENT_PATTERNS` and union all matched
       ``content_types`` (preserving order, deduplicating).
    """
    text = question or ""
    if not text.strip():
        return QueryFilters()

    matched: list[str] = []

    department: str | None = None
    for pat, name in DEPARTMENT_PATTERNS:
        if pat.search(text):
            department = name
            matched.append(f"dept:{name}")
            break

    faculty: str | None = None
    for pat, name in FACULTY_PATTERNS:
        if pat.search(text):
            faculty = name
            matched.append(f"faculty:{name}")
            break
    if department and not faculty:
        inferred = DEPARTMENT_TO_FACULTY.get(department)
        if inferred:
            faculty = inferred
            matched.append(f"faculty(inferred):{inferred}")

    course_code: str | None = None
    code_match = COURSE_CODE_RE.search(text)
    if code_match:
        prefix = code_match.group(1).upper()
        number = code_match.group(2)
        candidate = f"{prefix}{number}"
        if candidate not in _COURSE_CODE_BLOCKLIST:
            course_code = candidate
            matched.append(f"course_code:{candidate}")

    semester: int | None = None
    sem_match = SEMESTER_RE.search(text)
    if sem_match:
        try:
            value = int(sem_match.group(1))
            if 1 <= value <= 12:
                semester = value
                matched.append(f"semester:{value}")
        except ValueError:
            pass

    content_types: list[str] = []
    seen_ct: set[str] = set()
    for pat, ctypes in INTENT_PATTERNS:
        if pat.search(text):
            for ct in ctypes:
                if ct not in seen_ct:
                    seen_ct.add(ct)
                    content_types.append(ct)
            matched.append(f"intent:{','.join(ctypes)}")

    return QueryFilters(
        faculty=faculty,
        department=department,
        course_code=course_code,
        semester=semester,
        content_types=tuple(content_types),
        matched_terms=tuple(matched),
    )


__all__ = [
    "QueryFilters",
    "parse_query",
    "DEPARTMENT_PATTERNS",
    "FACULTY_PATTERNS",
    "INTENT_PATTERNS",
    "DEPARTMENT_TO_FACULTY",
    "COURSE_CODE_RE",
    "SEMESTER_RE",
]

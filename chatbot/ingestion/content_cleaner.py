"""
Content cleaning utilities for the ingestion pipeline.

Two complementary entry points exist here:

1. ``clean_html_to_text`` (legacy, kept for backwards compatibility) —
   BeautifulSoup based, drops noise containers (nav/footer/script/...)
   and returns a plain-text body, the ``<title>`` and the first ``<h1>``.
   Used by the generic web crawler and the OIBS portal scraper.

2. ``clean_plain_text`` / ``clean_html_to_document`` (Step 3.1 additions) —
   plain-text post-processor that:
     * extracts URLs / e-mails into separate metadata buckets,
     * strips them from the body so the LLM is not paying tokens for
       boilerplate links (per project lead decision: URLs live only in
       ``metadata.source_url`` / ``metadata.contact_emails``),
     * removes navigation/footer chrome that survived the HTML pass
       (e.g. cookie banners on ``acibadem.edu.tr`` and the menu strip
       at the bottom of every OIBS Bologna page),
     * normalises whitespace and drops accidentally repeated lines.

The plain-text cleaner is the only path that the OIBS Bologna scraper
can use, because that scraper bypasses HTML and reads ``page.innerText``
directly to keep its iframe handling simple.

Source-kind switches (``"obs"`` / ``"www"`` / ``"generic"``) tune the
chrome dictionaries to the patterns observed on each source. Adding
support for a new source means appending its specific noise lines to
``_NAV_CHROME_BY_SOURCE`` — the rest of the pipeline stays the same.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Iterable

from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Public regex helpers
# ---------------------------------------------------------------------------

# HTTP/HTTPS URL — anchored on the scheme and stops at whitespace, common
# punctuation that can wrap URLs in prose, and angle/quote brackets that
# show up when text was originally HTML.
URL_RE = re.compile(
    r"""(?xi)
    \b
    (?:https?://|www\.)            # scheme or bare-www
    [^\s<>"'()\[\]{}]+             # body of the URL
    """,
)

# Standalone domain reference — only matched when surrounded by space and
# followed by a TLD-shaped suffix. Conservative on purpose so we don't
# eat acronyms like "Ph.D".
BARE_DOMAIN_RE = re.compile(
    r"""(?xi)
    (?<![\w@/.])
    (?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.){1,3}
    (?:tr|com|edu|org|net|gov|info|io|co)
    (?:/[^\s<>"'()]*)?
    \b
    """,
)

EMAIL_RE = re.compile(
    r"""(?xi)
    \b[a-z0-9._%+-]+
    @
    [a-z0-9.-]+\.[a-z]{2,}
    \b
    """,
)

# Phone number — Turkish-friendly: optional country code, optional
# parentheses on area code, mixed separators. Used only for extraction;
# we keep phone numbers in the body because they are usually answer-
# carrying content (contact pages).
PHONE_RE = re.compile(
    r"""
    (?:\+?\d{1,3}[\s\-.])?       # optional country code
    (?:\(?\d{2,4}\)?[\s\-.])?    # optional area code
    \d{3}[\s\-.]?\d{2,4}[\s\-.]?\d{0,4}
    """,
    re.VERBOSE,
)

# Cleanup primitives used in several places.
_MULTI_SPACE_RE = re.compile(r"[ \t\u00a0]{2,}")
_TRIPLE_NEWLINE_RE = re.compile(r"\n{3,}")
_TRAILING_PUNCT_RE = re.compile(r"[\s.,;:•·\-—–]+$")
_LEADING_PUNCT_RE = re.compile(r"^[\s.,;:•·\-—–]+")

# ---------------------------------------------------------------------------
# Source-specific chrome dictionaries
# ---------------------------------------------------------------------------

# Lines that, when they appear *alone* (after trimming punctuation), are
# almost certainly navigation/footer leftovers. Matched case-insensitively
# against the full line content. Only exact-line matches are stripped —
# the same string in a sentence is preserved.
_GENERIC_NAV_CHROME: frozenset[str] = frozenset(
    {
        "anasayfa",
        "ana sayfa",
        "menu",
        "menü",
        "yardım",
        "yardim",
        "giriş yap",
        "giris yap",
        "login",
        "logout",
        "çıkış",
        "cikis",
        "türkçe",
        "turkce",
        "english",
        "tr",
        "en",
        "ara",
        "arama",
        "search",
        "tıklayın",
        "tiklayin",
        "tıkla",
        "tikla",
        "devamı",
        "devami",
        "daha fazla",
        "more",
        "kabul et",
        "tamam",
        "kapat",
        "geri",
        "ileri",
    }
)

# Cookie / KVKK / copyright noise that shows up in www.acibadem.edu.tr
# templates. Substring match (case-insensitive) — if the line contains
# any of these, drop the entire line.
_WWW_NOISE_SUBSTRINGS: tuple[str, ...] = (
    "çerez politikası",
    "cerez politikasi",
    "çerez kullanımı",
    "cerez kullanimi",
    "cookie policy",
    "kvkk",
    "kişisel verilerin korunması",
    "kisisel verilerin korunmasi",
    "tüm hakları saklıdır",
    "tum haklari saklidir",
    "all rights reserved",
    "© ",
    "copyright",
    "site haritası",
    "site haritasi",
    "sitemap",
    "sosyal medya",
    "social media",
)

# OIBS-Bologna leftovers that survive plain-text extraction. Each line
# is matched as an exact full-line equivalent (post normalisation).
_OBS_NAV_CHROME: frozenset[str] = frozenset(
    {
        "akademik birimler",
        "lisans",
        "lisansüstü",
        "lisansustu",
        "önlisans",
        "onlisans",
        "kişisel bilgiler",
        "kisisel bilgiler",
        "üniversite hayatı",
        "universite hayati",
        "önemli linkler",
        "onemli linkler",
        "ders programı",
        "ders programi",
        "duyurular",
        "haberler",
        "etkinlikler",
        "ana sayfa",
    }
)

_NAV_CHROME_BY_SOURCE: dict[str, frozenset[str]] = {
    "generic": _GENERIC_NAV_CHROME,
    "obs": _GENERIC_NAV_CHROME | _OBS_NAV_CHROME,
    "www": _GENERIC_NAV_CHROME,
}


# ---------------------------------------------------------------------------
# Legacy HTML cleaner (unchanged public API — used by crawler & obs_actions)
# ---------------------------------------------------------------------------

DROP_SELECTORS = (
    "script",
    "style",
    "noscript",
    "header",
    "footer",
    "nav",
    "aside",
    "form",
    "svg",
    ".menu",
    ".navbar",
    ".footer",
    ".header",
    ".breadcrumb",
    ".breadcrumbs",
)

WHITESPACE_RE = re.compile(r"\s+")


def _best_content_root(soup: BeautifulSoup):
    candidates = soup.select("main, article, section, div.content, div.main, div#content")
    if candidates:
        return max(candidates, key=lambda node: len(node.get_text(" ", strip=True)))
    return soup.body or soup


def clean_html_to_text(html: str, max_chars: int = 200_000) -> tuple[str, str, str]:
    """Backwards-compatible HTML→text helper.

    Returns ``(text, title, heading)``. New ingestion code should prefer
    :func:`clean_html_to_document`, which additionally strips URLs and
    runs the source-aware plain-text post-processor on the body.
    """
    soup = BeautifulSoup(html, "lxml")

    for selector in DROP_SELECTORS:
        for tag in soup.select(selector):
            tag.decompose()

    title = ""
    if soup.title and soup.title.string:
        title = soup.title.string.strip()

    root = _best_content_root(soup)
    heading = ""
    h1 = root.find("h1")
    if h1:
        heading = WHITESPACE_RE.sub(" ", h1.get_text(" ", strip=True)).strip()

    raw_blocks: list[str] = []
    seen_blocks: set[str] = set()
    for node in root.select("h1, h2, h3, p, li, td, th, div"):
        text = WHITESPACE_RE.sub(" ", node.get_text(" ", strip=True)).strip()
        if not text or len(text) < 25:
            continue
        if text in seen_blocks:
            continue
        seen_blocks.add(text)
        raw_blocks.append(text)

    combined = "\n".join(raw_blocks)
    combined = unicodedata.normalize("NFC", combined)
    combined = WHITESPACE_RE.sub(" ", combined).strip()
    if len(combined) > max_chars:
        combined = combined[:max_chars].strip()
    return combined, title, heading


def content_hash(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()


# ---------------------------------------------------------------------------
# Plain-text post-processor (Step 3.1)
# ---------------------------------------------------------------------------


@dataclass
class CleanedTextResult:
    """Output of :func:`clean_plain_text`.

    ``text`` is the cleaned body. ``urls`` and ``emails`` are the unique,
    insertion-ordered values pulled out of the body — the caller is
    expected to attach them to ``metadata.source_url`` /
    ``metadata.contact_emails`` instead of leaving them in the prose.
    """

    text: str
    urls: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)
    dropped_lines: int = 0


@dataclass
class CleanedDocument:
    """Output of :func:`clean_html_to_document`."""

    text: str
    title: str
    heading: str
    urls: list[str] = field(default_factory=list)
    emails: list[str] = field(default_factory=list)


def extract_urls(text: str) -> list[str]:
    """Return every URL-shaped token from ``text``, deduplicated and in
    first-seen order. Both ``http(s)://`` and bare ``www.`` forms are
    captured; bare domains ending in known TLDs are also picked up so
    that ``acibadem.edu.tr/contact`` shows up even without a scheme.
    """
    if not text:
        return []
    seen: dict[str, None] = {}
    for match in URL_RE.finditer(text):
        url = _normalise_url(match.group(0))
        if url:
            seen.setdefault(url, None)
    for match in BARE_DOMAIN_RE.finditer(text):
        url = _normalise_url(match.group(0))
        if url and url not in seen:
            seen[url] = None
    return list(seen.keys())


def extract_emails(text: str) -> list[str]:
    if not text:
        return []
    seen: dict[str, None] = {}
    for match in EMAIL_RE.finditer(text):
        seen.setdefault(match.group(0).lower(), None)
    return list(seen.keys())


def strip_urls(text: str, *, replace_with: str = " ") -> str:
    """Remove every URL-shaped token. URLs that occupy a whole line are
    handled at the line-level pass below (the empty line is dropped),
    while URLs embedded in prose collapse to a single space so the
    surrounding sentence stays grammatical.
    """
    if not text:
        return text
    text = URL_RE.sub(replace_with, text)
    text = BARE_DOMAIN_RE.sub(replace_with, text)
    return text


def strip_emails(text: str, *, replace_with: str = " ") -> str:
    """Email addresses are useful for metadata but uninformative inside
    the body — the LLM gains nothing from seeing the literal address.
    Mirroring the URL behaviour, we drop them out of the prose.
    """
    if not text:
        return text
    return EMAIL_RE.sub(replace_with, text)


def normalize_whitespace(text: str) -> str:
    """Canonicalise whitespace without flattening paragraph breaks.

    * ``\\r\\n`` / ``\\r`` → ``\\n``
    * NBSP / zero-width-space → space / removed
    * Tabs → single space (Bologna ``about`` page emits tab-separated cells)
    * 2+ consecutive spaces → 1 space
    * 3+ consecutive newlines → 2 newlines (paragraph boundary kept)
    * Per-line ``strip()`` so trailing tabs from OIBS don't leak into
      downstream chunkers.
    """
    if not text:
        return text
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ").replace("\u200b", "")
    text = text.replace("\t", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _TRIPLE_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def dedup_consecutive_lines(text: str) -> str:
    """Drop lines that are byte-identical to the line just before them.

    OIBS templates frequently repeat the page heading just under the
    main banner (e.g. ``Bilgisayar Mühendisliği`` appears on every
    sub-page). We only deduplicate immediate neighbours so legitimate
    repetition (lists, table-like rows) survives.
    """
    if not text:
        return text
    out: list[str] = []
    last_norm = ""
    for line in text.split("\n"):
        norm = line.strip().lower()
        if norm and norm == last_norm:
            continue
        out.append(line)
        last_norm = norm
    return "\n".join(out)


def strip_navigation_chrome(text: str, *, source_kind: str = "generic") -> tuple[str, int]:
    """Remove lines that consist entirely of navigation/footer chrome.

    Returns ``(cleaned_text, dropped_count)``. The dropped count is
    surfaced for observability — useful when tuning new source-kind
    dictionaries.
    """
    if not text:
        return text, 0
    chrome_set = _NAV_CHROME_BY_SOURCE.get(source_kind, _GENERIC_NAV_CHROME)
    use_www_substrings = source_kind == "www"

    kept: list[str] = []
    dropped = 0
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            kept.append(raw_line)
            continue
        cleaned_line = _LEADING_PUNCT_RE.sub("", _TRAILING_PUNCT_RE.sub("", line)).strip()
        lowered = cleaned_line.lower()
        if not lowered:
            dropped += 1
            continue
        if lowered in chrome_set:
            dropped += 1
            continue
        if use_www_substrings and any(s in lowered for s in _WWW_NOISE_SUBSTRINGS):
            dropped += 1
            continue
        # Single-character or single-word stub lines (e.g. lone "•" or
        # "›" that survived the HTML pass) are noise across all sources.
        if len(lowered) <= 2 and not lowered.isdigit():
            dropped += 1
            continue
        kept.append(raw_line)
    return "\n".join(kept), dropped


def clean_plain_text(
    text: str,
    *,
    source_kind: str = "generic",
    drop_emails: bool = True,
) -> CleanedTextResult:
    """End-to-end plain-text cleaner.

    Pipeline:
        1. Extract URLs / e-mails into the result (so the caller can
           write them to metadata).
        2. Strip URLs from the body. Optionally strip e-mails.
        3. Normalise whitespace.
        4. Strip per-source navigation/footer chrome.
        5. Drop consecutive duplicate lines.

    The function is idempotent: calling it twice on the same input
    yields the same output (URL/email lists become empty on the
    second pass because the body has nothing left to extract).
    """
    if not text:
        return CleanedTextResult(text="")

    urls = extract_urls(text)
    emails = extract_emails(text) if drop_emails else []

    body = strip_urls(text)
    if drop_emails:
        body = strip_emails(body)
    body = normalize_whitespace(body)
    body, dropped = strip_navigation_chrome(body, source_kind=source_kind)
    body = normalize_whitespace(body)
    body = dedup_consecutive_lines(body)
    body = normalize_whitespace(body)

    return CleanedTextResult(
        text=body,
        urls=urls,
        emails=emails,
        dropped_lines=dropped,
    )


def clean_html_to_document(
    html: str,
    *,
    source_kind: str = "www",
    max_chars: int = 200_000,
) -> CleanedDocument:
    """HTML→text wrapper that runs the plain-text post-processor.

    This is the entry point Step 3.3 will plug into the crawler so
    each ScrapedPage gets a body free of cookie banners and a clean
    list of URLs / e-mails captured from the original markup.
    """
    text, title, heading = clean_html_to_text(html, max_chars=max_chars)
    cleaned = clean_plain_text(text, source_kind=source_kind)
    return CleanedDocument(
        text=cleaned.text,
        title=title,
        heading=heading,
        urls=cleaned.urls,
        emails=cleaned.emails,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _normalise_url(raw: str) -> str:
    """Trim trailing punctuation that often follows a URL in prose
    (``)``, ``.``, ``,`` …) and prepend a default scheme to bare-www
    matches. Returns an empty string if the result no longer looks
    like a URL after trimming.
    """
    if not raw:
        return ""
    url = raw.strip().rstrip(".,;:!?)\"'»>")
    if not url:
        return ""
    if url.lower().startswith("www."):
        url = "https://" + url
    if not url.lower().startswith(("http://", "https://")):
        # Bare-domain match. Synthesise an https URL for storage so
        # downstream code can navigate to it without further work.
        url = "https://" + url
    return url


__all__ = (
    # Legacy API (do not break)
    "DROP_SELECTORS",
    "WHITESPACE_RE",
    "clean_html_to_text",
    "content_hash",
    # Step 3.1 additions
    "URL_RE",
    "EMAIL_RE",
    "PHONE_RE",
    "CleanedTextResult",
    "CleanedDocument",
    "extract_urls",
    "extract_emails",
    "strip_urls",
    "strip_emails",
    "normalize_whitespace",
    "dedup_consecutive_lines",
    "strip_navigation_chrome",
    "clean_plain_text",
    "clean_html_to_document",
)

from __future__ import annotations

import posixpath
from urllib.parse import ParseResult, parse_qsl, urlencode, urljoin, urlparse, urlunparse


# Project specification (PDF) names exactly two public sources:
#   - https://www.acibadem.edu.tr  (also reachable via the apex acibadem.edu.tr)
#   - https://obs.acibadem.edu.tr
# We deliberately do NOT accept arbitrary acibadem.edu.tr subdomains so that the
# "public-data only" boundary is enforced at the URL layer instead of relying on
# downstream filters.
ALLOWED_HOSTS: frozenset[str] = frozenset(
    {
        "www.acibadem.edu.tr",
        "acibadem.edu.tr",
        "obs.acibadem.edu.tr",
    }
)

SKIP_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".rar",
    ".7z",
    ".mp3",
    ".mp4",
    ".avi",
    ".mov",
    ".css",
    ".js",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
}

# Login / authenticated areas — the project specification forbids logging in.
# Both English and Turkish slugs are listed because Acibadem pages mix locales.
# IMPORTANT: announcements ("/duyurular") and contact ("/iletisim") are NOT
# blocked here; the project specification requires them in the corpus.
SKIP_PATH_HINTS = {
    "/login",
    "/signin",
    "/sign-in",
    "/auth",
    "/account",
    "/logout",
    "/admin",
    "/wp-admin",
    "/wp-login",
    "/giris",
    "/giris-yap",
    "/oturum",
    "/oturum-ac",
    "/uye-girisi",
    "/uyegirisi",
    "/uye-ol",
    "/kayit-ol",
    "/parola",
    "/sifre",
    "/sifremi-unuttum",
    "/hesap",
    "/cikis",
}

# Some sites carry login intent in query parameters (e.g. ``?action=login``)
# rather than in the URL path. We block these too.
SKIP_QUERY_TOKENS = {
    "login",
    "signin",
    "logout",
    "giris",
    "oturum",
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if not parsed.scheme:
        parsed = urlparse(f"https://{url.strip()}")
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    path = posixpath.normpath(path)
    if not path.startswith("/"):
        path = f"/{path}"
    if parsed.path.endswith("/") and not path.endswith("/"):
        path += "/"
    query = urlencode(sorted(parse_qsl(parsed.query, keep_blank_values=False)))
    normalized = ParseResult(
        scheme=parsed.scheme.lower(),
        netloc=netloc,
        path=path,
        params="",
        query=query,
        fragment="",
    )
    return urlunparse(normalized)


def resolve_link(base_url: str, href: str) -> str | None:
    if not href or href.startswith("#") or href.startswith("mailto:") or href.startswith("tel:"):
        return None
    return normalize_url(urljoin(base_url, href))


def is_allowed_domain(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in ALLOWED_HOSTS


def source_type_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    return "obs" if host == "obs.acibadem.edu.tr" else "main_site"


def should_skip_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    lowered_path = parsed.path.lower()
    for ext in SKIP_EXTENSIONS:
        if lowered_path.endswith(ext):
            return True, f"asset_extension:{ext}"
    for blocked in SKIP_PATH_HINTS:
        # Match the slug as a path segment to avoid false positives like
        # "/about/<account-of-...>"; we still allow ``in`` for prefix variants.
        if blocked in lowered_path:
            return True, f"restricted_path_hint:{blocked}"
    if parsed.query:
        query_lower = parsed.query.lower()
        for token in SKIP_QUERY_TOKENS:
            if token in query_lower:
                return True, f"restricted_query_token:{token}"
    return False, ""


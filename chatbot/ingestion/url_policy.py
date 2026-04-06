from __future__ import annotations

import posixpath
from urllib.parse import ParseResult, parse_qsl, urlencode, urljoin, urlparse, urlunparse


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

SKIP_PATH_HINTS = {
    "/login",
    "/signin",
    "/auth",
    "/account",
    "/logout",
    "/admin",
    "/wp-admin",
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
    return host.endswith("acibadem.edu.tr") or host == "obs.acibadem.edu.tr"


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
        if blocked in lowered_path:
            return True, f"restricted_path_hint:{blocked}"
    return False, ""


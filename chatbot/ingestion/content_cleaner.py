from __future__ import annotations

import re
import unicodedata
from hashlib import sha256

from bs4 import BeautifulSoup

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


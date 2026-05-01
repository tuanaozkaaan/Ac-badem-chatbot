from __future__ import annotations

import re
import time
import random
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests
from bs4 import BeautifulSoup

ALLOWED_DOMAINS = {"www.acibadem.edu.tr", "acibadem.edu.tr", "obs.acibadem.edu.tr"}
MAX_PAGES = 50
REQUEST_TIMEOUT_SECONDS = 15
DELAY_MIN_SECONDS = 1.5
DELAY_MAX_SECONDS = 2.0

SEED_URLS = [
    "https://www.acibadem.edu.tr/",
    "https://www.acibadem.edu.tr/akademik/",
    "https://www.acibadem.edu.tr/kayit/",
    "https://www.acibadem.edu.tr/yasam/",
    "https://www.acibadem.edu.tr/uluslararasi/",
    "https://www.acibadem.edu.tr/iletisim/",
    "https://www.acibadem.edu.tr/burslar/",
    "https://www.acibadem.edu.tr/ucretler/",
    "https://obs.acibadem.edu.tr/",
]

SKIP_PATH_KEYWORDS = (
    "login",
    "giris",
    "signin",
    "auth",
    "hesap",
    "account",
    "oturum",
    "wp-admin",
    "cart",
    "checkout",
)
SKIP_EXTENSIONS = (
    ".pdf",
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".zip",
    ".rar",
    ".mp4",
    ".mp3",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
)


def normalize_url(url: str) -> str:
    clean, _frag = urldefrag(url.strip())
    parsed = urlparse(clean)
    # Keep query only for OBS pages where public program params may matter.
    query = parsed.query if parsed.netloc == "obs.acibadem.edu.tr" else ""
    path = parsed.path.rstrip("/") or "/"
    return parsed._replace(path=path, query=query).geturl()


def is_allowed_url(url: str) -> bool:
    try:
        p = urlparse(url)
    except Exception:
        return False
    if p.scheme not in {"http", "https"}:
        return False
    if p.netloc not in ALLOWED_DOMAINS:
        return False
    lower_url = url.lower()
    if any(lower_url.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    if any(k in lower_url for k in SKIP_PATH_KEYWORDS):
        return False
    return True


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def remove_layout_noise(soup: BeautifulSoup) -> None:
    for tag in soup.select("script, style, noscript, svg, form, iframe"):
        tag.decompose()
    for tag in soup.select("nav, footer, header, aside"):
        tag.decompose()
    noisy_nodes = []
    for tag in soup.find_all(True):
        attrs = getattr(tag, "attrs", None) or {}
        cls = " ".join(attrs.get("class", [])).lower() if isinstance(attrs.get("class"), list) else str(attrs.get("class", "")).lower()
        tid = str(attrs.get("id", "")).lower()
        if any(k in cls for k in ("nav", "menu", "footer", "breadcrumb", "cookie", "social")):
            noisy_nodes.append(tag)
            continue
        if any(k in tid for k in ("nav", "menu", "footer", "breadcrumb", "cookie", "social")):
            noisy_nodes.append(tag)
    for tag in noisy_nodes:
        try:
            tag.decompose()
        except Exception:
            pass


def extract_content(soup: BeautifulSoup) -> str:
    blocks: list[str] = []

    # Prioritize main/article content when available.
    container = soup.select_one("main") or soup.select_one("article") or soup.body or soup
    for el in container.find_all(["h1", "h2", "h3", "h4", "p", "li", "table"]):
        if el.name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [clean_text(td.get_text(" ", strip=True)) for td in tr.find_all(["th", "td"])]
                cells = [c for c in cells if c]
                if cells:
                    rows.append(" | ".join(cells))
            if rows:
                blocks.append("Table:")
                blocks.extend(rows)
            continue
        txt = clean_text(el.get_text(" ", strip=True))
        if not txt:
            continue
        # Skip very short boilerplate-like snippets.
        if len(txt) < 2:
            continue
        blocks.append(txt)

    # Deduplicate consecutive repeats.
    deduped: list[str] = []
    prev = ""
    for b in blocks:
        if b == prev:
            continue
        deduped.append(b)
        prev = b
    return "\n".join(deduped).strip()


def safe_filename(title: str, url: str, used: set[str]) -> str:
    base = clean_text(title)
    if not base:
        p = urlparse(url)
        base = p.path.strip("/").split("/")[-1] or p.netloc
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", base).strip("_").lower()
    slug = slug[:80] or "page"
    name = f"{slug}.txt"
    i = 2
    while name in used:
        name = f"{slug}_{i}.txt"
        i += 1
    used.add(name)
    return name


def crawl() -> int:
    out_dir = Path("data") / "scraped"
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "AcibademChatbotScraper/1.0 (+public educational indexing; contact: local project)",
            "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8",
        }
    )

    queue = deque(normalize_url(u) for u in SEED_URLS)
    visited: set[str] = set()
    saved_count = 0
    used_names: set[str] = set()

    print(f"Starting crawl with max pages={MAX_PAGES}")

    while queue and saved_count < MAX_PAGES:
        url = queue.popleft()
        if url in visited:
            print(f"SKIPPED (already visited): {url}")
            continue
        if not is_allowed_url(url):
            print(f"SKIPPED (filtered): {url}")
            continue
        visited.add(url)
        print(f"VISITED: {url}")

        try:
            resp = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            if resp.status_code != 200:
                print(f"SKIPPED (status={resp.status_code}): {url}")
                continue
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" not in content_type:
                print(f"SKIPPED (non-html): {url}")
                continue
        except Exception as exc:
            print(f"ERROR (request): {url} -> {exc}")
            continue

        try:
            soup = BeautifulSoup(resp.text, "lxml")
            remove_layout_noise(soup)
            page_title = clean_text((soup.title.get_text(" ", strip=True) if soup.title else "").strip())
            body_text = extract_content(soup)
            if len(body_text) < 80:
                print(f"SKIPPED (low content): {url}")
            else:
                filename = safe_filename(page_title, url, used_names)
                output_path = out_dir / filename
                output_path.write_text(
                    f"Source URL: {url}\n\n"
                    f"Page Title: {page_title or 'Untitled'}\n\n"
                    "Content:\n"
                    f"{body_text}\n",
                    encoding="utf-8",
                )
                saved_count += 1
                print(f"SAVED: {output_path}")

            for a in soup.find_all("a", href=True):
                href = a.get("href", "").strip()
                if not href:
                    continue
                absolute = normalize_url(urljoin(url, href))
                if absolute in visited:
                    continue
                if not is_allowed_url(absolute):
                    continue
                queue.append(absolute)
        except Exception as exc:
            print(f"ERROR (parse): {url} -> {exc}")
        finally:
            # Responsible rate limiting between requests.
            time.sleep(random.uniform(DELAY_MIN_SECONDS, DELAY_MAX_SECONDS))

    print(f"TOTAL_PAGES_SAVED={saved_count}")
    return saved_count


if __name__ == "__main__":
    crawl()

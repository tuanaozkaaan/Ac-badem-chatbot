from __future__ import annotations

import logging
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    html: str
    final_url: str
    status_code: int
    used_playwright: bool = False


class Fetcher:
    def __init__(self, user_agent: str, timeout_seconds: int) -> None:
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "tr-TR,tr;q=0.9,en;q=0.8"})

    def fetch_requests(self, url: str) -> FetchResult:
        response = self.session.get(url, timeout=self.timeout_seconds, allow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("Content-Type", "").lower()
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            raise ValueError(f"non_html_content:{content_type}")
        response.encoding = response.encoding or "utf-8"
        return FetchResult(html=response.text, final_url=response.url, status_code=response.status_code)

    def fetch_playwright(self, url: str) -> FetchResult:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("playwright_unavailable") from exc

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(user_agent=self.session.headers["User-Agent"])
            response = page.goto(url, wait_until="networkidle", timeout=self.timeout_seconds * 1000)
            html = page.content()
            final_url = page.url
            status_code = response.status if response else 200
            browser.close()
            return FetchResult(
                html=html,
                final_url=final_url,
                status_code=status_code,
                used_playwright=True,
            )

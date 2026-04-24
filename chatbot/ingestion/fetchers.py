from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

OBS_DEBUG_HTML = Path("/tmp/obs_debug.html")
OBS_DEBUG_TXT = Path("/tmp/obs_debug.txt")


@dataclass
class FetchResult:
    html: str
    final_url: str
    status_code: int
    used_playwright: bool = False
    # Playwright: body inner_text from main + iframes (see crawler merge).
    rendered_plain_text: str | None = None
    playwright_page_title: str | None = None
    # OBS: postback/click ile açılan ek ScrapedPage adayları ve doğrudan href/window.open URL'leri.
    obs_captures: list[Any] | None = None
    obs_seed_urls: list[str] = field(default_factory=list)


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

    def fetch_playwright(
        self,
        url: str,
        *,
        explore_obs_actions: bool = False,
        obs_max_action_clicks: int = 20,
    ) -> FetchResult:
        try:
            from playwright.sync_api import sync_playwright
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("playwright_unavailable") from exc

        timeout_ms = int(self.timeout_seconds * 1000)
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            try:
                page = browser.new_page(user_agent=self.session.headers["User-Agent"])
                response = page.goto(url, wait_until="networkidle", timeout=timeout_ms)
                page.wait_for_timeout(3000)
                html = page.content()
                try:
                    pw_title = (page.title() or "").strip() or None
                except Exception:
                    pw_title = None

                rendered_parts: list[str] = []
                seen_plain: set[str] = set()
                inner_timeout = min(15_000, timeout_ms)
                for frame in page.frames:
                    try:
                        loc = frame.locator("body")
                        if loc.count() == 0:
                            continue
                        chunk = loc.inner_text(timeout=inner_timeout)
                        s = (chunk or "").strip()
                        if s and s not in seen_plain:
                            seen_plain.add(s)
                            rendered_parts.append(s)
                    except Exception:
                        continue
                rendered_plain = "\n\n".join(rendered_parts) if rendered_parts else None
                # explore sonrası page.url değişebilir; ana kayıt bu yükleme anına ait kalsın
                final_url = page.url
                status_code = response.status if response else 200

                try:
                    OBS_DEBUG_HTML.write_text(html or "", encoding="utf-8", errors="replace")
                    OBS_DEBUG_TXT.write_text(rendered_plain or "", encoding="utf-8", errors="replace")
                except OSError as wexc:
                    logger.debug("obs_debug_file_write_failed: %s", wexc)

                obs_caps: list[Any] | None = None
                obs_seeds: list[str] = []
                if explore_obs_actions and "obs.acibadem.edu.tr" in (url or "").lower():
                    from chatbot.ingestion.obs_actions import explore_obs_action_pages

                    try:
                        caps, seeds = explore_obs_action_pages(
                            page, url, timeout_ms, int(obs_max_action_clicks)
                        )
                        obs_caps = caps or None
                        obs_seeds = list(seeds or [])
                    except Exception as exc:
                        logger.warning("OBS explore_failed url=%s err=%s", url, exc)
                        obs_caps = None
                        obs_seeds = []

                return FetchResult(
                    html=html,
                    final_url=final_url,
                    status_code=status_code,
                    used_playwright=True,
                    rendered_plain_text=rendered_plain,
                    playwright_page_title=pw_title,
                    obs_captures=obs_caps,
                    obs_seed_urls=obs_seeds,
                )
            finally:
                browser.close()

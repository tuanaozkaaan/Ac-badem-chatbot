from __future__ import annotations

import heapq
import logging
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

from .config import CrawlConfig
from .content_cleaner import clean_html_to_text, content_hash
from .fetchers import Fetcher, FetchResult
from .storage import upsert_page
from .url_policy import is_allowed_domain, normalize_url, resolve_link, should_skip_url, source_type_for_url

logger = logging.getLogger(__name__)

OBS_HOST_MARKER = "obs.acibadem.edu.tr"


@dataclass
class CrawlStats:
    visited: int = 0
    fetched: int = 0
    stored_created: int = 0
    stored_updated: int = 0
    skipped: int = 0
    failed: int = 0


class ResponsibleCrawler:
    def __init__(self, config: CrawlConfig) -> None:
        self.config = config
        self.stats = CrawlStats()
        self.fetcher = Fetcher(user_agent=config.user_agent, timeout_seconds=config.timeout_seconds)
        self.visited: set[str] = set()
        self.enqueued: set[str] = set()
        self.robots_by_domain: dict[str, RobotFileParser] = {}
        self.last_request_ts: dict[str, float] = defaultdict(float)
        self._queue: list[tuple[int, int, str]] = []
        self._counter = 0

    def _text_from_fetch(self, fetch_result: FetchResult) -> tuple[str, str, str]:
        """HTML + optional Playwright frame inner_text birleşimi; başlık için page.title tercihi."""
        text, title, section = clean_html_to_text(
            fetch_result.html, max_chars=self.config.max_content_chars
        )
        plain = (fetch_result.rendered_plain_text or "").strip()
        if plain:
            base = text.strip()
            merged = f"{base}\n\n---\n\n{plain}".strip() if base else plain
            if len(merged) > self.config.max_content_chars:
                merged = merged[: self.config.max_content_chars]
            text = merged
        if fetch_result.used_playwright and fetch_result.playwright_page_title:
            pt = fetch_result.playwright_page_title.strip()
            if pt and (not (title or "").strip() or len(pt) > len((title or "").strip())):
                title = pt[:512]
        return text, title, section

    def crawl(self) -> CrawlStats:
        for seed in self.config.seed_urls:
            self._enqueue(seed)

        while self._queue and self.stats.fetched < self.config.max_pages:
            _, _, current_url = heapq.heappop(self._queue)
            current_url = normalize_url(current_url)
            if current_url in self.visited:
                continue

            self.visited.add(current_url)
            self.stats.visited += 1

            should_skip, skip_reason = should_skip_url(current_url)
            if should_skip:
                self.stats.skipped += 1
                logger.info("SKIP url=%s reason=%s", current_url, skip_reason)
                continue

            if not self._can_fetch_by_robots(current_url):
                self.stats.skipped += 1
                logger.info("SKIP url=%s reason=robots_disallow", current_url)
                continue

            self._apply_delay(urlparse(current_url).netloc.lower())

            source_guess = source_type_for_url(current_url)
            fetch_result = None
            explore_obs = (
                source_guess == "obs"
                and self.config.enable_playwright_for_obs
                and self.config.obs_max_action_clicks > 0
            )

            # OBS: istenirse tam sayfa render (JS) — ScrapedPage için önce Playwright
            if source_guess == "obs" and self.config.enable_playwright_for_obs:
                try:
                    fetch_result = self.fetcher.fetch_playwright(
                        current_url,
                        explore_obs_actions=explore_obs,
                        obs_max_action_clicks=self.config.obs_max_action_clicks,
                    )
                    logger.info("INFO url=%s used=playwright_obs_primary", current_url)
                except Exception as exc:
                    logger.warning(
                        "WARN playwright_obs_primary_failed url=%s error=%s (trying HTTP)",
                        current_url,
                        exc,
                    )

            if fetch_result is None:
                try:
                    fetch_result = self.fetcher.fetch_requests(current_url)
                except Exception as exc:
                    self.stats.failed += 1
                    logger.warning("FAIL fetch url=%s error=%s", current_url, exc)
                    continue

            html = fetch_result.html
            source_type = source_type_for_url(fetch_result.final_url)
            text, title, section = self._text_from_fetch(fetch_result)
            # Playwright başarısız olduysa veya bayrak kapalıyken ince gövde: bir kez Playwright dene
            if (
                source_type == "obs"
                and self.config.enable_playwright_for_obs
                and not fetch_result.used_playwright
                and len(text) < 300
            ):
                try:
                    fetch_result = self.fetcher.fetch_playwright(
                        current_url,
                        explore_obs_actions=explore_obs,
                        obs_max_action_clicks=self.config.obs_max_action_clicks,
                    )
                    html = fetch_result.html
                    text, title, section = self._text_from_fetch(fetch_result)
                    logger.info("INFO url=%s used=playwright_obs_thin_retry", current_url)
                except Exception as exc:
                    logger.warning("WARN playwright_fallback_failed url=%s error=%s", current_url, exc)

            final_url_l = (fetch_result.final_url or "").lower()
            is_obs_host = OBS_HOST_MARKER in final_url_l
            min_text_len = 35 if (is_obs_host and fetch_result.used_playwright) else 120

            if len(text) < min_text_len:
                if is_obs_host:
                    logger.warning(
                        "OBS low_content_debug url=%s text_len=%d title=%r first_500=%r",
                        fetch_result.final_url,
                        len(text or ""),
                        title,
                        (text or "")[:500],
                    )
                self.stats.skipped += 1
                logger.info(
                    "SKIP url=%s reason=low_content min_len=%d text_len=%d",
                    current_url,
                    min_text_len,
                    len(text or ""),
                )
                continue

            norm_url = normalize_url(fetch_result.final_url)
            result = upsert_page(
                url=norm_url,
                url_variant="",
                title=title[:512],
                section=section[:256],
                source_type=source_type,
                content=text,
                content_hash=content_hash(text),
            )
            self.stats.fetched += 1
            if result.action == "created":
                self.stats.stored_created += 1
            elif result.action == "updated":
                self.stats.stored_updated += 1
            else:
                self.stats.skipped += 1
            logger.info("STORE url=%s action=%s page_id=%s", current_url, result.action, result.page_id)

            parent_ch = content_hash(text)
            for cap in fetch_result.obs_captures or []:
                if self.stats.fetched >= self.config.max_pages:
                    break
                cch = content_hash(cap.content)
                if cch == parent_ch:
                    continue
                if len(cap.content.strip()) < 40:
                    continue
                r2 = upsert_page(
                    url=normalize_url(cap.canonical_url),
                    url_variant=cap.url_variant[:128],
                    title=cap.title[:512],
                    section=cap.section[:256],
                    source_type=source_type,
                    content=cap.content,
                    content_hash=cch,
                )
                self.stats.fetched += 1
                if r2.action == "created":
                    self.stats.stored_created += 1
                elif r2.action == "updated":
                    self.stats.stored_updated += 1
                else:
                    self.stats.skipped += 1
                logger.info(
                    "STORE_OBS_ACTION url=%s variant=%s action=%s page_id=%s",
                    cap.canonical_url,
                    cap.url_variant[:16],
                    r2.action,
                    r2.page_id,
                )

            for link in self._extract_links(fetch_result.final_url, html):
                self._enqueue(link)
            if is_obs_host:
                for su in fetch_result.obs_seed_urls or []:
                    self._enqueue(normalize_url(su))
                for su in self._extract_obs_urls_from_html(html):
                    self._enqueue(su)

        logger.info(
            "CRAWL_DONE visited=%d fetched=%d created=%d updated=%d skipped=%d failed=%d",
            self.stats.visited,
            self.stats.fetched,
            self.stats.stored_created,
            self.stats.stored_updated,
            self.stats.skipped,
            self.stats.failed,
        )
        return self.stats

    def _enqueue(self, url: str) -> None:
        normalized = normalize_url(url)
        if normalized in self.enqueued or normalized in self.visited:
            return
        if not is_allowed_domain(normalized):
            logger.info("SKIP url=%s reason=outside_domain", normalized)
            return
        should_skip, reason = should_skip_url(normalized)
        if should_skip:
            logger.info("SKIP url=%s reason=%s", normalized, reason)
            return
        priority = self._priority_for_url(normalized)
        self._counter += 1
        heapq.heappush(self._queue, (priority, self._counter, normalized))
        self.enqueued.add(normalized)

    def _priority_for_url(self, url: str) -> int:
        lowered = url.lower()
        score = 0
        for keyword in self.config.high_value_keywords:
            if keyword in lowered:
                score += 1
        return -score

    def _extract_links(self, base_url: str, html: str) -> set[str]:
        soup = BeautifulSoup(html, "lxml")
        links: set[str] = set()
        for a in soup.select("a[href]"):
            resolved = resolve_link(base_url, a.get("href", ""))
            if not resolved:
                continue
            if is_allowed_domain(resolved):
                links.add(resolved)
        return links

    def _extract_obs_urls_from_html(self, html: str) -> set[str]:
        """Statik HTML içindeki window.open / location.href ile geçen obs.acibadem.edu.tr URL'leri."""
        out: set[str] = set()
        if not html or OBS_HOST_MARKER not in html.lower():
            return out
        for m in re.finditer(
            r"(?:window\.open|location\.href)\s*\(\s*['\"](https?://[^'\"]*obs\.acibadem\.edu\.tr[^'\"]*)",
            html,
            re.I,
        ):
            u = normalize_url(m.group(1).strip())
            if is_allowed_domain(u):
                out.add(u)
        return out

    def _can_fetch_by_robots(self, url: str) -> bool:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        robots = self.robots_by_domain.get(domain)
        if not robots:
            robots_url = f"{parsed.scheme}://{domain}/robots.txt"
            robots = RobotFileParser()
            robots.set_url(robots_url)
            try:
                robots.read()
            except Exception as exc:
                logger.warning("WARN robots_read_failed url=%s error=%s", robots_url, exc)
            self.robots_by_domain[domain] = robots
        return robots.can_fetch(self.config.user_agent, url)

    def _apply_delay(self, domain: str) -> None:
        now = time.time()
        elapsed = now - self.last_request_ts[domain]
        min_wait = random.uniform(self.config.min_delay_seconds, self.config.max_delay_seconds)
        if elapsed < min_wait:
            time.sleep(min_wait - elapsed)
        self.last_request_ts[domain] = time.time()

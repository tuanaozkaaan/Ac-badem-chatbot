from __future__ import annotations

import heapq
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup

from .config import CrawlConfig
from .content_cleaner import clean_html_to_text, content_hash
from .fetchers import Fetcher
from .storage import upsert_page
from .url_policy import is_allowed_domain, normalize_url, resolve_link, should_skip_url, source_type_for_url

logger = logging.getLogger(__name__)


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

            try:
                fetch_result = self.fetcher.fetch_requests(current_url)
            except Exception as exc:
                self.stats.failed += 1
                logger.warning("FAIL fetch url=%s error=%s", current_url, exc)
                continue

            html = fetch_result.html
            source_type = source_type_for_url(fetch_result.final_url)
            text, title, section = clean_html_to_text(html, max_chars=self.config.max_content_chars)
            if (
                source_type == "obs"
                and self.config.enable_playwright_for_obs
                and len(text) < 300
            ):
                try:
                    fetch_result = self.fetcher.fetch_playwright(current_url)
                    html = fetch_result.html
                    text, title, section = clean_html_to_text(
                        html, max_chars=self.config.max_content_chars
                    )
                    logger.info("INFO url=%s used=playwright", current_url)
                except Exception as exc:
                    logger.warning("WARN playwright_fallback_failed url=%s error=%s", current_url, exc)

            if len(text) < 120:
                self.stats.skipped += 1
                logger.info("SKIP url=%s reason=low_content", current_url)
                continue

            result = upsert_page(
                url=normalize_url(fetch_result.final_url),
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

            for link in self._extract_links(fetch_result.final_url, html):
                self._enqueue(link)

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

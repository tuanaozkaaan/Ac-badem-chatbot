from __future__ import annotations

import logging

from django.core.management.base import BaseCommand, CommandParser

from chatbot.ingestion.config import CrawlConfig
from chatbot.ingestion.crawler import ResponsibleCrawler
from chatbot.ingestion.url_policy import normalize_url


# High-value seeds that satisfy the project specification:
#   - General university pages (faculties, campus life, scholarships, fees)
#   - Mandatory: announcements ("/duyurular") and contact ("/iletisim")
#   - OBS Bologna entry point
# Link discovery will expand from here; explicit seeds make the crawl
# deterministic when ``--max-pages`` is small (e.g. CI smoke runs).
DEFAULT_SEED_URLS: tuple[str, ...] = (
    "https://www.acibadem.edu.tr/",
    "https://www.acibadem.edu.tr/akademik/",
    "https://www.acibadem.edu.tr/akademik/fakulteler/",
    "https://www.acibadem.edu.tr/akademik/yuksekokullar/",
    "https://www.acibadem.edu.tr/akademik/enstituler/",
    "https://www.acibadem.edu.tr/kayit/",
    "https://www.acibadem.edu.tr/burslar/",
    "https://www.acibadem.edu.tr/ucretler/",
    "https://www.acibadem.edu.tr/yasam/",
    "https://www.acibadem.edu.tr/uluslararasi/",
    "https://www.acibadem.edu.tr/akademik-takvim/",
    # Spec-mandatory: must be in the corpus even if noisy.
    "https://www.acibadem.edu.tr/iletisim/",
    "https://www.acibadem.edu.tr/duyurular/",
    # OBS Bologna public entry; deeper pilot crawl is configured in step 2.
    "https://obs.acibadem.edu.tr/",
)


class Command(BaseCommand):
    help = "Crawl Acibadem public pages responsibly and store cleaned raw pages in PostgreSQL."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--seed-url",
            action="append",
            default=[],
            help=(
                "Seed URL to crawl (can be provided multiple times). "
                "When omitted, a curated list of high-value Acibadem pages is used."
            ),
        )
        parser.add_argument("--max-pages", type=int, default=200)
        parser.add_argument("--min-delay", type=float, default=1.0)
        parser.add_argument("--max-delay", type=float, default=2.0)
        parser.add_argument("--timeout", type=int, default=20)
        parser.add_argument("--enable-playwright-obs", action="store_true")
        parser.add_argument(
            "--obs-max-action-clicks",
            type=int,
            default=20,
            help="OBS Playwright: tıklanacak en fazla aksiyon (0=postback keşfi kapalı).",
        )
        parser.add_argument("--log-level", default="INFO")

    def handle(self, *args, **options):
        self._setup_logging(options["log_level"])
        seed_urls = options["seed_url"] or list(DEFAULT_SEED_URLS)
        config = CrawlConfig(
            seed_urls=[normalize_url(url) for url in seed_urls],
            max_pages=options["max_pages"],
            min_delay_seconds=options["min_delay"],
            max_delay_seconds=options["max_delay"],
            timeout_seconds=options["timeout"],
            enable_playwright_for_obs=options["enable_playwright_obs"],
            obs_max_action_clicks=options["obs_max_action_clicks"],
        )
        crawler = ResponsibleCrawler(config)
        stats = crawler.crawl()

        self.stdout.write(
            self.style.SUCCESS(
                "Ingestion completed: "
                f"visited={stats.visited}, fetched={stats.fetched}, "
                f"created={stats.stored_created}, updated={stats.stored_updated}, "
                f"skipped={stats.skipped}, failed={stats.failed}"
            )
        )

    def _setup_logging(self, level: str) -> None:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )

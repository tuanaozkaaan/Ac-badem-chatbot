from __future__ import annotations

import logging

from django.core.management.base import BaseCommand, CommandParser

from chatbot.ingestion.config import CrawlConfig
from chatbot.ingestion.crawler import ResponsibleCrawler
from chatbot.ingestion.url_policy import normalize_url


class Command(BaseCommand):
    help = "Crawl Acibadem public pages responsibly and store cleaned raw pages in PostgreSQL."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--seed-url",
            action="append",
            default=[],
            help="Seed URL to crawl (can be provided multiple times).",
        )
        parser.add_argument("--max-pages", type=int, default=200)
        parser.add_argument("--min-delay", type=float, default=1.0)
        parser.add_argument("--max-delay", type=float, default=2.0)
        parser.add_argument("--timeout", type=int, default=20)
        parser.add_argument("--enable-playwright-obs", action="store_true")
        parser.add_argument("--log-level", default="INFO")

    def handle(self, *args, **options):
        self._setup_logging(options["log_level"])
        seed_urls = options["seed_url"] or [
            "https://www.acibadem.edu.tr",
            "https://obs.acibadem.edu.tr",
        ]
        config = CrawlConfig(
            seed_urls=[normalize_url(url) for url in seed_urls],
            max_pages=options["max_pages"],
            min_delay_seconds=options["min_delay"],
            max_delay_seconds=options["max_delay"],
            timeout_seconds=options["timeout"],
            enable_playwright_for_obs=options["enable_playwright_obs"],
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

from __future__ import annotations

import logging

from django.core.management.base import BaseCommand, CommandParser

from chatbot.chunking.service import ChunkingConfig, chunk_single_page
from chatbot.models import ScrapedPage


class Command(BaseCommand):
    help = "Generate retrieval-friendly text chunks from existing ScrapedPage rows."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--page-id", type=int, action="append", default=[], help="Process only specific page IDs.")
        parser.add_argument(
            "--source-type",
            choices=[ScrapedPage.SOURCE_MAIN_SITE, ScrapedPage.SOURCE_OBS],
            help="Filter pages by source type.",
        )
        parser.add_argument("--limit", type=int, default=0, help="Optional max number of pages to process.")
        parser.add_argument("--force", action="store_true", help="Re-chunk pages even if content did not change.")
        parser.add_argument("--dry-run", action="store_true", help="Preview chunk counts without writing to database.")

        parser.add_argument("--chunk-size", type=int, default=1000, help="Max chars per chunk.")
        parser.add_argument("--overlap", type=int, default=180, help="Overlap chars between chunks.")
        parser.add_argument("--min-chars", type=int, default=120, help="Minimum chars required for a valid chunk.")
        parser.add_argument("--min-words", type=int, default=20, help="Minimum words required for a valid chunk.")
        parser.add_argument("--max-chunks-per-page", type=int, default=200, help="Safety cap for chunks per page.")
        parser.add_argument("--log-level", default="INFO")

    def handle(self, *args, **options):
        self._setup_logging(options["log_level"])

        chunk_size = options["chunk_size"]
        overlap = options["overlap"]
        if overlap >= chunk_size:
            raise ValueError("--overlap must be smaller than --chunk-size")

        config = ChunkingConfig(
            chunk_size_chars=chunk_size,
            overlap_chars=overlap,
            min_chunk_chars=options["min_chars"],
            min_word_count=options["min_words"],
            max_chunks_per_page=options["max_chunks_per_page"],
        )

        pages = ScrapedPage.objects.all().order_by("id")
        page_ids = options["page_id"] or []
        if page_ids:
            pages = pages.filter(id__in=page_ids)
        if options.get("source_type"):
            pages = pages.filter(source_type=options["source_type"])
        if options["limit"] and options["limit"] > 0:
            pages = pages[: options["limit"]]

        processed_pages = 0
        skipped_pages = 0
        metadata_refreshed_pages = 0
        chunk_count = 0

        for page in pages.iterator(chunk_size=50):
            result = chunk_single_page(
                page,
                config=config,
                force=options["force"],
                dry_run=options["dry_run"],
            )

            if result.action == "skipped":
                skipped_pages += 1
                continue
            if result.action == "metadata_refreshed":
                # Content unchanged, but ScrapedPage.metadata was updated
                # by a later ingest run; chunks just got their copy
                # synced down. No chunks were rebuilt.
                metadata_refreshed_pages += 1
                continue

            processed_pages += 1
            chunk_count += result.chunks_created

        mode = "DRY RUN" if options["dry_run"] else "WRITE"
        self.stdout.write(
            self.style.SUCCESS(
                f"Chunking completed [{mode}] - processed_pages={processed_pages}, "
                f"skipped_pages={skipped_pages}, "
                f"metadata_refreshed_pages={metadata_refreshed_pages}, "
                f"total_chunks={chunk_count}"
            )
        )

    def _setup_logging(self, level: str) -> None:
        logging.basicConfig(
            level=getattr(logging, level.upper(), logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        )

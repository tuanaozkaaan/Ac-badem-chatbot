from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from django.core.management.base import BaseCommand, CommandParser
from django.db import transaction
from django.utils import timezone

from chatbot.models import PageChunk, ScrapedPage


_WHITESPACE_RE = re.compile(r"\s+")


def _normalize_ws(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass(frozen=True)
class ChunkingSpec:
    # Target chunk size. For long documents this yields multiple 300-500 word chunks.
    chunk_words: int = 400
    overlap_words: int = 50
    # Keep this low enough to allow a short final "tail" chunk to be stored.
    min_chunk_words: int = 60


def _word_chunks(text: str, *, spec: ChunkingSpec) -> list[str]:
    """
    Split text into word-based chunks with overlap.
    Target chunk size ~300-500 words (defaults to 400) and overlap ~50 words.
    """
    if spec.overlap_words >= spec.chunk_words:
        raise ValueError("overlap_words must be smaller than chunk_words")

    cleaned = _normalize_ws(text)
    if not cleaned:
        return []

    words = cleaned.split(" ")
    total = len(words)
    if total == 0:
        return []

    # We never store the whole file as a single chunk.
    # If the document is shorter than the target chunk size, split it into 2+ smaller chunks
    # (best-effort) so retrieval has multiple focused passages to match.
    if total <= spec.chunk_words:
        overlap = min(spec.overlap_words, max(0, total // 10))
        if overlap >= total:
            overlap = 0
        # Aim for 2-4 chunks depending on length.
        parts = 2 if total < 2 * (spec.min_chunk_words + 1) else 3
        step = max(1, (total - overlap) // parts)
        out: list[str] = []
        start = 0
        idx_guard = 0
        while start < total and idx_guard < 10:
            end = min(total, start + step + overlap)
            chunk_words = words[start:end]
            if chunk_words:
                out.append(" ".join(chunk_words).strip())
            if end >= total:
                break
            start = max(0, end - overlap)
            idx_guard += 1
        # De-dup any accidental identical chunks.
        deduped: list[str] = []
        for c in out:
            if c and (not deduped or c != deduped[-1]):
                deduped.append(c)
        return deduped

    step = spec.chunk_words - spec.overlap_words
    out: list[str] = []
    start = 0
    while start < len(words):
        chunk_words = words[start : start + spec.chunk_words]
        # Always keep meaningful chunks; allow a shorter final tail chunk.
        if len(chunk_words) >= spec.min_chunk_words:
            out.append(" ".join(chunk_words).strip())
        start += step
        # If we're close to the end and the next window would be too small, stop.
        if start >= len(words):
            break

    # Ensure we always include the tail if it wasn't captured.
    if out:
        last_end = (len(out) - 1) * step + spec.chunk_words
        if last_end < len(words):
            tail = words[max(0, len(words) - spec.chunk_words) : len(words)]
            if len(tail) >= spec.min_chunk_words:
                tail_text = " ".join(tail).strip()
                if tail_text and tail_text != out[-1]:
                    out.append(tail_text)
    return out


class Command(BaseCommand):
    help = "Ingest /app/data/*.txt into ScrapedPage + PageChunk (clears existing rows first)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--data-dir",
            default="/app/data",
            help="Directory containing .txt files (default: /app/data).",
        )
        parser.add_argument(
            "--chunk-words",
            type=int,
            default=400,
            help="Target words per chunk (default: 400).",
        )
        parser.add_argument(
            "--overlap-words",
            type=int,
            default=50,
            help="Overlap words between chunks (default: 50).",
        )
        parser.add_argument(
            "--min-chunk-words",
            type=int,
            default=60,
            help="Minimum words required for a chunk to be stored (default: 60).",
        )

    def handle(self, *args, **options):
        data_dir = Path(str(options["data_dir"])).expanduser()
        spec = ChunkingSpec(
            chunk_words=int(options["chunk_words"]),
            overlap_words=int(options["overlap_words"]),
            min_chunk_words=int(options["min_chunk_words"]),
        )

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        if not data_dir.is_dir():
            raise NotADirectoryError(f"Not a directory: {data_dir}")

        txt_files = sorted(data_dir.glob("*.txt"))
        if not txt_files:
            self.stdout.write(self.style.WARNING(f"No .txt files found in {data_dir}"))
            return

        with transaction.atomic():
            files_processed = 0
            chunks_created = 0
            now = timezone.now()

            for file_path in txt_files:
                raw = file_path.read_text(encoding="utf-8", errors="replace")
                content = raw.strip()
                if not content:
                    continue

                # Use stable, Docker-safe pseudo-URL so URLField uniqueness is satisfied.
                # (We don't have a real URL for local text files.)
                pseudo_url = f"file://{file_path.resolve()}"
                title = file_path.stem.replace("_", " ").strip()
                section = "local_txt"
                tags: list[str] = []
                if "contact" in file_path.stem.lower():
                    tags = ["contact", "address", "iletişim", "adres"]
                    title = "İletişim / Adres"
                    section = "contact_address"
                    # Put tags up-front to increase retrieval recall.
                    content = f"Etiketler: {', '.join(tags)}\n\n{content}"
                source_type = ScrapedPage.SOURCE_MAIN_SITE
                content_hash = _sha256(content)

                # Upsert strategy for local files:
                # - delete any prior ScrapedPage with same pseudo_url (cascades to chunks)
                # - recreate with updated content/hash
                ScrapedPage.objects.filter(url=pseudo_url).delete()

                page = ScrapedPage.objects.create(
                    url=pseudo_url,
                    title=title,
                    section=section,
                    source_type=source_type,
                    content=content,
                    content_hash=content_hash,
                    crawled_at=now,
                )

                chunks = _word_chunks(content, spec=spec)
                # For contact/address docs we want to keep the whole address together as a single chunk.
                if tags and len(chunks) > 1:
                    chunks = ["\n".join([c.strip() for c in chunks if c.strip()]).strip()]
                if not chunks:
                    files_processed += 1
                    continue

                chunk_rows: list[PageChunk] = []
                for idx, chunk_text in enumerate(chunks):
                    chunk_hash = _sha256(chunk_text)
                    chunk_rows.append(
                        PageChunk(
                            scraped_page=page,
                            chunk_index=idx,
                            chunk_text=chunk_text,
                            chunk_hash=chunk_hash,
                            page_content_hash=content_hash,
                            title=title,
                            section=section,
                            source_type=source_type,
                            url=pseudo_url,
                            char_count=len(chunk_text),
                            token_count_estimate=max(1, len(chunk_text) // 4),
                        )
                    )

                PageChunk.objects.bulk_create(chunk_rows, batch_size=500)
                files_processed += 1
                chunks_created += len(chunk_rows)
                self.stdout.write(f"{file_path.name}: chunks_created={len(chunk_rows)}")

        self.stdout.write(
            self.style.SUCCESS(
                f"ingest_txt_data completed: files_processed={files_processed}, chunks_created={chunks_created}"
            )
        )

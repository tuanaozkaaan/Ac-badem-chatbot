from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from hashlib import sha256
from typing import Iterable

from django.db import transaction

from chatbot.models import PageChunk, ScrapedPage

WHITESPACE_RE = re.compile(r"[ \t]+")
EXCESS_NEWLINES_RE = re.compile(r"\n{3,}")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9ÇĞİÖŞÜ])")
HEADING_LINE_RE = re.compile(
    r"^(#{1,6}\s+.+|[A-Z0-9ÇĞİÖŞÜ][A-Z0-9ÇĞİÖŞÜ \-:]{5,}|(\d+(\.\d+)*)\s+.+)$"
)


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size_chars: int = 1000
    overlap_chars: int = 180
    min_chunk_chars: int = 120
    min_word_count: int = 20
    max_chunks_per_page: int = 200


@dataclass(frozen=True)
class ChunkingResult:
    action: str
    page_id: int
    chunks_created: int
    reason: str = ""


def _sha256(text: str) -> str:
    return sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text or "")
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    normalized = WHITESPACE_RE.sub(" ", normalized)
    normalized = EXCESS_NEWLINES_RE.sub("\n\n", normalized)
    return normalized.strip()


def _split_semantic_sections(text: str) -> list[str]:
    """
    Prefer heading-aware blocks if the input has line structure.
    Falls back to a single section when structure is weak.
    """
    if not text:
        return []

    lines = [line.strip() for line in text.split("\n")]
    if len(lines) <= 1:
        return [text]

    sections: list[str] = []
    current: list[str] = []
    heading_count = 0

    for line in lines:
        if not line:
            if current:
                current.append("")
            continue

        if HEADING_LINE_RE.match(line):
            heading_count += 1
            if current:
                section = "\n".join(current).strip()
                if section:
                    sections.append(section)
            current = [line]
            continue

        current.append(line)

    if current:
        section = "\n".join(current).strip()
        if section:
            sections.append(section)

    # If we detected no meaningful heading structure, use single block.
    if heading_count == 0:
        return [text]
    return sections or [text]


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    parts = SENTENCE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _hard_split_long_sentence(sentence: str, chunk_size_chars: int) -> list[str]:
    """
    Fallback for very long sentences: split on word boundaries by size.
    """
    words = sentence.split()
    if not words:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        add_len = len(word) + (1 if current else 0)
        if current and current_len + add_len > chunk_size_chars:
            chunks.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += add_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(sentences: list[str], overlap_chars: int) -> list[str]:
    if overlap_chars <= 0:
        return []
    selected: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        sentence_len = len(sentence) + (1 if selected else 0)
        if selected and total + sentence_len > overlap_chars:
            break
        selected.append(sentence)
        total += sentence_len
    selected.reverse()
    return selected


def _build_chunks(section_texts: Iterable[str], config: ChunkingConfig) -> list[str]:
    chunks: list[str] = []

    for section_text in section_texts:
        sentences = _split_sentences(section_text)
        if not sentences:
            continue

        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            if len(sentence) > config.chunk_size_chars:
                split_parts = _hard_split_long_sentence(sentence, config.chunk_size_chars)
            else:
                split_parts = [sentence]

            for part in split_parts:
                add_len = len(part) + (1 if current else 0)

                if current and current_len + add_len > config.chunk_size_chars:
                    chunks.append(" ".join(current).strip())
                    overlap = _overlap_tail(current, config.overlap_chars)
                    current = overlap[:] if overlap else []
                    current_len = sum(len(s) for s in current) + max(0, len(current) - 1)

                    add_len = len(part) + (1 if current else 0)
                    if current and current_len + add_len > config.chunk_size_chars:
                        chunks.append(" ".join(current).strip())
                        current = []
                        current_len = 0

                current.append(part)
                current_len += len(part) + (1 if len(current) > 1 else 0)

        if current:
            chunks.append(" ".join(current).strip())

    # Drop empty and duplicate generated chunks while preserving order.
    deduped: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        if not chunk:
            continue
        if chunk in seen:
            continue
        seen.add(chunk)
        deduped.append(chunk)
    return deduped


def _is_low_value_chunk(text: str, config: ChunkingConfig) -> bool:
    if not text:
        return True
    if len(text) < config.min_chunk_chars:
        return True

    words = text.split()
    if len(words) < config.min_word_count:
        return True

    alnum_chars = sum(1 for c in text if c.isalnum())
    if not alnum_chars:
        return True
    ratio = alnum_chars / max(len(text), 1)
    return ratio < 0.45


def _estimate_token_count(text: str) -> int:
    # Practical approximation: about 4 characters per token for mixed TR/EN content.
    return max(1, round(len(text) / 4))


def generate_chunks_for_content(content: str, config: ChunkingConfig) -> list[str]:
    normalized = _normalize_text(content)
    if not normalized:
        return []

    sections = _split_semantic_sections(normalized)
    raw_chunks = _build_chunks(sections, config)

    filtered: list[str] = []
    for chunk in raw_chunks:
        if _is_low_value_chunk(chunk, config):
            continue
        filtered.append(chunk)
        if len(filtered) >= config.max_chunks_per_page:
            break
    return filtered


@transaction.atomic
def chunk_single_page(
    page: ScrapedPage,
    config: ChunkingConfig,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> ChunkingResult:
    existing_qs = PageChunk.objects.filter(scraped_page=page)
    has_up_to_date_chunks = existing_qs.filter(page_content_hash=page.content_hash).exists()
    if not force and has_up_to_date_chunks:
        return ChunkingResult(
            action="skipped",
            page_id=page.id,
            chunks_created=0,
            reason="unchanged_content",
        )

    chunks = generate_chunks_for_content(page.content, config)
    if not chunks:
        if dry_run:
            return ChunkingResult(action="dry_run", page_id=page.id, chunks_created=0, reason="no_chunks")
        existing_qs.delete()
        return ChunkingResult(action="processed", page_id=page.id, chunks_created=0, reason="no_chunks")

    new_rows: list[PageChunk] = []
    seen_hashes: set[str] = set()
    for idx, chunk_text in enumerate(chunks):
        chunk_hash = _sha256(chunk_text)
        if chunk_hash in seen_hashes:
            continue
        seen_hashes.add(chunk_hash)
        new_rows.append(
            PageChunk(
                scraped_page=page,
                chunk_index=idx,
                chunk_text=chunk_text,
                chunk_hash=chunk_hash,
                page_content_hash=page.content_hash,
                title=page.title,
                section=page.section,
                source_type=page.source_type,
                url=page.url,
                char_count=len(chunk_text),
                token_count_estimate=_estimate_token_count(chunk_text),
            )
        )

    if dry_run:
        return ChunkingResult(
            action="dry_run",
            page_id=page.id,
            chunks_created=len(new_rows),
        )

    existing_qs.delete()
    PageChunk.objects.bulk_create(new_rows, batch_size=200)
    return ChunkingResult(
        action="processed",
        page_id=page.id,
        chunks_created=len(new_rows),
    )

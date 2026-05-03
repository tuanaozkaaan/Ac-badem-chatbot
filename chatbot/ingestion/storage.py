"""
ScrapedPage upsert helper.

The crawler and the OBS Bologna pilot share this single entry point so
idempotency rules live in one place:

  * ``(url, url_variant)`` is the natural key for a row — calling
    :func:`upsert_page` again with the same pair UPDATEs the existing
    row instead of creating a duplicate.
  * ``content_hash`` is unique across the table; if the same body
    arrives under a different ``(url, url_variant)`` we report a
    ``skipped_duplicate_hash`` instead of corrupting the existing row.
  * ``metadata`` is overwritten on update (we always trust the latest
    ingestion to produce the most accurate metadata blob).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.utils import timezone

from chatbot.models import ScrapedPage


@dataclass
class StoreResult:
    action: str
    page_id: int | None = None
    reason: str = ""


def upsert_page(
    *,
    url: str,
    title: str,
    section: str,
    source_type: str,
    content: str,
    content_hash: str,
    url_variant: str = "",
    metadata: dict[str, Any] | None = None,
) -> StoreResult:
    variant = (url_variant or "").strip()[:128]
    payload = dict(metadata or {})

    existing_by_hash = ScrapedPage.objects.filter(content_hash=content_hash).first()
    existing_row = ScrapedPage.objects.filter(url=url, url_variant=variant).first()

    if existing_by_hash and (not existing_row or existing_by_hash.id != existing_row.id):
        return StoreResult(
            action="skipped_duplicate_hash",
            page_id=existing_by_hash.id,
            reason="duplicate_content",
        )

    now = timezone.now()
    if existing_row:
        existing_row.title = title
        existing_row.section = section
        existing_row.source_type = source_type
        existing_row.content = content
        existing_row.content_hash = content_hash
        existing_row.metadata = payload
        existing_row.crawled_at = now
        existing_row.save(
            update_fields=[
                "title",
                "section",
                "source_type",
                "content",
                "content_hash",
                "metadata",
                "crawled_at",
                "updated_at",
            ]
        )
        return StoreResult(action="updated", page_id=existing_row.id)

    new_page = ScrapedPage.objects.create(
        url=url,
        url_variant=variant,
        title=title,
        section=section,
        source_type=source_type,
        content=content,
        content_hash=content_hash,
        metadata=payload,
        crawled_at=now,
    )
    return StoreResult(action="created", page_id=new_page.id)

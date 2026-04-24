from __future__ import annotations

from dataclasses import dataclass

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
) -> StoreResult:
    variant = (url_variant or "").strip()[:128]
    existing_by_hash = ScrapedPage.objects.filter(content_hash=content_hash).first()
    existing_row = ScrapedPage.objects.filter(url=url, url_variant=variant).first()

    if existing_by_hash and (not existing_row or existing_by_hash.id != existing_row.id):
        return StoreResult(action="skipped_duplicate_hash", page_id=existing_by_hash.id, reason="duplicate_content")

    now = timezone.now()
    if existing_row:
        existing_row.title = title
        existing_row.section = section
        existing_row.source_type = source_type
        existing_row.content = content
        existing_row.content_hash = content_hash
        existing_row.crawled_at = now
        existing_row.save(
            update_fields=[
                "title",
                "section",
                "source_type",
                "content",
                "content_hash",
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
        crawled_at=now,
    )
    return StoreResult(action="created", page_id=new_page.id)

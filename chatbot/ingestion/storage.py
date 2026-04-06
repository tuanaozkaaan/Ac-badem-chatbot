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
) -> StoreResult:
    existing_by_hash = ScrapedPage.objects.filter(content_hash=content_hash).first()
    existing_by_url = ScrapedPage.objects.filter(url=url).first()

    if existing_by_hash and (not existing_by_url or existing_by_hash.id != existing_by_url.id):
        return StoreResult(action="skipped_duplicate_hash", page_id=existing_by_hash.id, reason="duplicate_content")

    now = timezone.now()
    if existing_by_url:
        existing_by_url.title = title
        existing_by_url.section = section
        existing_by_url.source_type = source_type
        existing_by_url.content = content
        existing_by_url.content_hash = content_hash
        existing_by_url.crawled_at = now
        existing_by_url.save(
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
        return StoreResult(action="updated", page_id=existing_by_url.id)

    new_page = ScrapedPage.objects.create(
        url=url,
        title=title,
        section=section,
        source_type=source_type,
        content=content,
        content_hash=content_hash,
        crawled_at=now,
    )
    return StoreResult(action="created", page_id=new_page.id)

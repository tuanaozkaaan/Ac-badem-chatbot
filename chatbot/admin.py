from django.contrib import admin

from .models import PageChunk, ScrapedPage


@admin.register(ScrapedPage)
class ScrapedPageAdmin(admin.ModelAdmin):
    list_display = ("source_type", "url", "title", "content_hash", "crawled_at")
    search_fields = ("url", "title", "section", "content_hash")
    list_filter = ("source_type", "crawled_at")


@admin.register(PageChunk)
class PageChunkAdmin(admin.ModelAdmin):
    list_display = (
        "scraped_page_id",
        "chunk_index",
        "source_type",
        "section",
        "char_count",
        "token_count_estimate",
        "updated_at",
    )
    search_fields = ("url", "title", "section", "chunk_hash")
    list_filter = ("source_type", "section", "updated_at")

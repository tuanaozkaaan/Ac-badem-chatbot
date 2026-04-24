from django.contrib import admin

from .models import Conversation, Message, PageChunk, ScrapedPage


@admin.register(ScrapedPage)
class ScrapedPageAdmin(admin.ModelAdmin):
    list_display = ("source_type", "url", "title", "content_hash", "crawled_at")
    search_fields = ("url", "title", "section", "content_hash")
    list_filter = ("source_type", "crawled_at")


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "created_at", "updated_at")
    search_fields = ("title",)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation_id", "role", "created_at")
    list_filter = ("role",)
    search_fields = ("content",)


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

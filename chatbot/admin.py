from django.contrib import admin

from .models import ScrapedPage


@admin.register(ScrapedPage)
class ScrapedPageAdmin(admin.ModelAdmin):
    list_display = ("source_type", "url", "title", "content_hash", "crawled_at")
    search_fields = ("url", "title", "section", "content_hash")
    list_filter = ("source_type", "crawled_at")

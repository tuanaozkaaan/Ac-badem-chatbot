from django.db import models


class ScrapedPage(models.Model):
    SOURCE_MAIN_SITE = "main_site"
    SOURCE_OBS = "obs"
    SOURCE_CHOICES = (
        (SOURCE_MAIN_SITE, "Main Site"),
        (SOURCE_OBS, "OBS"),
    )

    url = models.URLField(max_length=1024, unique=True, db_index=True)
    title = models.CharField(max_length=512, blank=True)
    section = models.CharField(max_length=256, blank=True)
    source_type = models.CharField(max_length=16, choices=SOURCE_CHOICES, db_index=True)
    content = models.TextField()
    content_hash = models.CharField(max_length=64, unique=True, db_index=True)
    crawled_at = models.DateTimeField(auto_now=True, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-crawled_at",)

    def __str__(self) -> str:
        return f"{self.source_type}: {self.url}"

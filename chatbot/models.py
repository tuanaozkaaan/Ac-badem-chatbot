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


class PageChunk(models.Model):
    scraped_page = models.ForeignKey(
        ScrapedPage,
        on_delete=models.CASCADE,
        related_name="chunks",
        db_index=True,
    )
    chunk_index = models.PositiveIntegerField()
    chunk_text = models.TextField()
    chunk_hash = models.CharField(max_length=64, db_index=True)
    page_content_hash = models.CharField(max_length=64, db_index=True)

    # Denormalized metadata for easier retrieval filtering/debugging.
    title = models.CharField(max_length=512, blank=True)
    section = models.CharField(max_length=256, blank=True)
    source_type = models.CharField(max_length=16, choices=ScrapedPage.SOURCE_CHOICES, db_index=True)
    url = models.URLField(max_length=1024, db_index=True)
    char_count = models.PositiveIntegerField(default=0)
    token_count_estimate = models.PositiveIntegerField(default=0)

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("scraped_page_id", "chunk_index")
        constraints = [
            models.UniqueConstraint(fields=["scraped_page", "chunk_index"], name="uniq_page_chunk_index"),
            models.UniqueConstraint(fields=["scraped_page", "chunk_hash"], name="uniq_page_chunk_hash"),
        ]
        indexes = [
            models.Index(fields=["source_type", "section"]),
            models.Index(fields=["url"]),
        ]

    def __str__(self) -> str:
        return f"PageChunk(page_id={self.scraped_page_id}, idx={self.chunk_index})"


class Conversation(models.Model):
    """UI chat session persisted for history across reloads."""

    title = models.CharField(max_length=200, blank=True, default="")
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ("-updated_at",)

    def __str__(self) -> str:
        return self.title or f"Conversation {self.pk}"


class Message(models.Model):
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_CHOICES = (
        (ROLE_USER, "User"),
        (ROLE_ASSISTANT, "Assistant"),
    )

    conversation = models.ForeignKey(
        Conversation,
        on_delete=models.CASCADE,
        related_name="messages",
        db_index=True,
    )
    role = models.CharField(max_length=16, choices=ROLE_CHOICES, db_index=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ("created_at",)
        indexes = [
            models.Index(fields=["conversation", "created_at"]),
        ]

    def __str__(self) -> str:
        return f"Message({self.role}, conv={self.conversation_id})"


class ChunkEmbedding(models.Model):
    chunk = models.OneToOneField(
        PageChunk,
        on_delete=models.CASCADE,
        related_name="embedding_data",
        db_index=True,
    )
    # Vektörleri JSON formatında (liste olarak) saklayacağız
    vector = models.JSONField() 
    embedding_model = models.CharField(max_length=255, default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dim = models.PositiveIntegerField()
    chunk_hash = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"Embedding for Chunk {self.chunk_id}"
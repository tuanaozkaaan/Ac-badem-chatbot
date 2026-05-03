"""
Add ``metadata`` JSONField to both ``ScrapedPage`` and ``PageChunk``.

This is part of Step 3.3 of the data-ingestion plan. The field stores
the structured metadata produced by ``chatbot.ingestion.metadata_enricher``
(faculty, department, content_type, course_code, semester, related_urls,
contact_emails, ...) so the RAG retriever can apply per-row filters at
search time.

Default ``{}`` ensures the migration is non-destructive: existing rows
get an empty dict, future ingestions overwrite it through the upsert
path. ``blank=True`` keeps the admin form happy.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0009_conversation_session_key"),
    ]

    operations = [
        migrations.AddField(
            model_name="scrapedpage",
            name="metadata",
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name="pagechunk",
            name="metadata",
            field=models.JSONField(blank=True, default=dict),
        ),
    ]

from django.db import migrations, models


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name="ScrapedPage",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("url", models.URLField(db_index=True, max_length=1024, unique=True)),
                ("title", models.CharField(blank=True, max_length=512)),
                ("section", models.CharField(blank=True, max_length=256)),
                (
                    "source_type",
                    models.CharField(
                        choices=[("main_site", "Main Site"), ("obs", "OBS")], db_index=True, max_length=16
                    ),
                ),
                ("content", models.TextField()),
                ("content_hash", models.CharField(db_index=True, max_length=64, unique=True)),
                ("crawled_at", models.DateTimeField(auto_now=True, db_index=True)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
                ("updated_at", models.DateTimeField(auto_now=True)),
            ],
            options={"ordering": ("-crawled_at",)},
        ),
    ]

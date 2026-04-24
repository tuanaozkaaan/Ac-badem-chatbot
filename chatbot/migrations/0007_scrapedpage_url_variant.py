from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0006_conversation_message"),
    ]

    operations = [
        migrations.AlterField(
            model_name="scrapedpage",
            name="url",
            field=models.URLField(db_index=True, max_length=1024),
        ),
        migrations.AddField(
            model_name="scrapedpage",
            name="url_variant",
            field=models.CharField(blank=True, db_index=True, default="", max_length=128),
        ),
        migrations.AddConstraint(
            model_name="scrapedpage",
            constraint=models.UniqueConstraint(
                fields=("url", "url_variant"),
                name="uniq_scrapedpage_url_url_variant",
            ),
        ),
    ]

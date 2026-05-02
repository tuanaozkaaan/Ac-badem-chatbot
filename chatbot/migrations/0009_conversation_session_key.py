from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("chatbot", "0008_rename_chatbot_mes_conv_created_idx_chatbot_mes_convers_b353f0_idx"),
    ]

    operations = [
        migrations.AddField(
            model_name="conversation",
            name="session_key",
            field=models.CharField(blank=True, db_index=True, default="", max_length=64),
        ),
    ]

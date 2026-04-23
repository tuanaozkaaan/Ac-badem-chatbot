from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Alias for ingest_txt_data (ingest /app/data/*.txt into DB)."

    def handle(self, *args, **options):
        from django.core.management import call_command

        call_command("ingest_txt_data")


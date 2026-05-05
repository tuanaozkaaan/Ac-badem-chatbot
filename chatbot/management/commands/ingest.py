from __future__ import annotations

from django.core.management import call_command
from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = (
        "Alias for ingest_txt_data: recursively ingest all ``.txt`` files under "
        "the corpus root (default: project ``data/``, i.e. ``/app/data`` in Docker)."
    )

    def add_arguments(self, parser) -> None:
        from chatbot.management.commands.ingest_txt_data import Command as IngestTxtCommand

        IngestTxtCommand().add_arguments(parser)

    def handle(self, *args, **options) -> None:
        call_command(
            "ingest_txt_data",
            *args,
            **options,
            stdout=self.stdout,
            stderr=self.stderr,
        )

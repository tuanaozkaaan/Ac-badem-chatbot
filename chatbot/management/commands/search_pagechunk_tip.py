"""
Quick DB check: how many PageChunk rows mention 'Tıp' (e.g. Tıp Fakültesi data present).
Run: python manage.py search_pagechunk_tip
"""

from __future__ import annotations

from django.core.management.base import BaseCommand
from django.db.models import Q

from chatbot.models import PageChunk


class Command(BaseCommand):
    help = "Count PageChunk rows whose chunk_text contains 'Tıp' (medical faculty sanity check)."

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--limit",
            type=int,
            default=8,
            help="How many sample rows to print (default: 8).",
        )

    def handle(self, *args, **options) -> None:
        limit: int = options["limit"]
        needle = "Tıp"
        qs = PageChunk.objects.filter(chunk_text__icontains=needle).order_by("id")
        total = qs.count()
        self.stdout.write(
            self.style.NOTICE(
                f"PageChunk rows with {needle!r} in chunk_text: {total}"
            )
        )
        if total == 0:
            self.stdout.write(
                self.style.WARNING(
                    "No matches. Ingested content may not include medical faculty text, "
                    "or text may use different spelling."
                )
            )
        else:
            for row in qs[:limit]:
                snippet = (row.chunk_text or "")[:220].replace("\n", " ")
                self.stdout.write(
                    f"  id={row.id} title={row.title!r} section={row.section!r}\n"
                    f"    url={row.url}\n"
                    f"    snippet: {snippet}..."
                )
            if total > limit:
                self.stdout.write(f"  ... and {total - limit} more row(s).")

        bilgisayar_qs = PageChunk.objects.filter(
            Q(chunk_text__icontains="Bilgisayar Mühendisliği")
            | Q(chunk_text__icontains="Bilgisayar Muhendisligi")
        ).order_by("id")
        bilgisayar_total = bilgisayar_qs.count()
        self.stdout.write(
            self.style.NOTICE(
                f"Bilgisayar Muhendisligi rows in chunk_text: {bilgisayar_total}"
            )
        )
        if bilgisayar_total == 0:
            self.stdout.write(
                self.style.WARNING(
                    "Bilgi Eksik: Veritabaninda 'Bilgisayar Muhendisligi' ifadesi bulunamadi."
                )
            )
            return

        first_row = bilgisayar_qs.first()
        if first_row is None:
            return
        sample = (first_row.chunk_text or "")[:220].replace("\n", " ")
        self.stdout.write(
            f"  sample id={first_row.id} title={first_row.title!r} section={first_row.section!r}\n"
            f"    url={first_row.url}\n"
            f"    snippet: {sample}..."
        )

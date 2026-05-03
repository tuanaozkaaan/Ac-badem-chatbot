"""Force-rebuild every ``ChunkEmbedding`` from scratch.

This is the operator-facing entry point for embedding-model migrations
(e.g. switching from ``all-MiniLM-L6-v2`` to
``paraphrase-multilingual-MiniLM-L12-v2``). It deliberately differs from
``create_embeddings`` in two ways:

  1. There is no incremental path. The table is cleared and every
     ``PageChunk`` is re-encoded with the target model. No silent skip,
     no risk of leaving rows from a prior model alive.
  2. The user-visible default is "yes, wipe everything"; ``create_embeddings``
     keeps an incremental default for the routine "I just chunked some new
     pages" workflow.

Internally this command is a thin wrapper over ``create_embeddings`` with
``--force`` baked in, so the encoding pipeline lives in exactly one place.
"""

from __future__ import annotations

from django.core.management import call_command
from django.core.management.base import BaseCommand

from rag.document_loader import EXPECTED_EMBEDDING_MODEL


class Command(BaseCommand):
    help = (
        "Tüm ChunkEmbedding satırlarını siler ve PageChunk'ların hepsini "
        "yeni embedding modeli ile baştan vektörleştirir. "
        "Embedding modeli değiştirildiğinde çalıştırılması zorunludur."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            help="SentenceTransformer.encode batch boyutu.",
        )
        parser.add_argument(
            "--write-batch-size",
            type=int,
            default=100,
            help="ChunkEmbedding.bulk_create batch boyutu.",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default=EXPECTED_EMBEDDING_MODEL,
            help=(
                "Embed model adını override et. Varsayılan: "
                f"{EXPECTED_EMBEDDING_MODEL!r} (rag.document_loader)."
            ),
        )

    def handle(self, *args, **options) -> None:
        self.stdout.write(
            self.style.WARNING(
                "rebuild_embeddings: tüm ChunkEmbedding satırları silinip "
                "yeniden üretilecek. (Force rebuild)"
            )
        )
        call_command(
            "create_embeddings",
            force=True,
            batch_size=options["batch_size"],
            write_batch_size=options["write_batch_size"],
            model_name=options["model_name"],
            stdout=self.stdout,
            stderr=self.stderr,
        )

"""Generate embeddings for ``PageChunk`` rows.

Default behaviour is *incremental*: only chunks without a matching
``ChunkEmbedding`` row are encoded, so this command is safe to run after
each chunking pass.

To swap the embedding model (e.g. moving from ``all-MiniLM-L6-v2`` to
``paraphrase-multilingual-MiniLM-L12-v2``) you MUST clear the existing
rows first — vectors from a different model live in a different semantic
space and silently produce garbage retrieval results. Either run

    python manage.py rebuild_embeddings

(which always force-resets) or pass ``--force`` to this command. A safety
guard refuses to add new rows whose declared model would diverge from
existing rows; it instructs the operator to use ``--force``.
"""

from __future__ import annotations

import logging
import time
from typing import Iterable

import numpy as np
from django.core.management.base import BaseCommand, CommandError

from chatbot.models import ChunkEmbedding, PageChunk
from rag.document_loader import EXPECTED_EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "PageChunk verilerini vektörize eder ve ChunkEmbedding tablosuna "
        "kaydeder. Yeni bir embedding modeline geçerken --force ile "
        "tablonun tamamı sıfırdan üretilmelidir."
    )

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--batch-size",
            type=int,
            default=64,
            help="SentenceTransformer.encode için tek seferde işlenecek chunk sayısı.",
        )
        parser.add_argument(
            "--write-batch-size",
            type=int,
            default=100,
            help="ChunkEmbedding.bulk_create grup büyüklüğü.",
        )
        parser.add_argument(
            "--force",
            action="store_true",
            help="Önce TÜM ChunkEmbedding satırlarını siler (clean-slate rebuild).",
        )
        parser.add_argument(
            "--model-name",
            type=str,
            default=EXPECTED_EMBEDDING_MODEL,
            help=(
                "Override embedding model name. Varsayılan: "
                f"{EXPECTED_EMBEDDING_MODEL!r}."
            ),
        )

    def handle(self, *args, **options) -> None:
        batch_size: int = max(1, int(options["batch_size"]))
        write_batch_size: int = max(1, int(options["write_batch_size"]))
        force: bool = bool(options["force"])
        model_name: str = options["model_name"].strip() or EXPECTED_EMBEDDING_MODEL

        if model_name != EXPECTED_EMBEDDING_MODEL:
            self.stdout.write(
                self.style.WARNING(
                    f"Model override aktif: {model_name!r} "
                    f"(EXPECTED_EMBEDDING_MODEL = {EXPECTED_EMBEDDING_MODEL!r})."
                )
            )

        if force:
            deleted, _ = ChunkEmbedding.objects.all().delete()
            self.stdout.write(
                self.style.WARNING(f"--force: {deleted} eski ChunkEmbedding satırı silindi.")
            )
        else:
            self._guard_against_model_mismatch(model_name)

        chunks_qs = self._select_chunks_to_process(model_name)
        total = chunks_qs.count()
        if total == 0:
            self.stdout.write(
                self.style.SUCCESS(
                    "İşlenecek yeni chunk yok. "
                    f"(PageChunk={PageChunk.objects.count()}, "
                    f"ChunkEmbedding={ChunkEmbedding.objects.count()})"
                )
            )
            self._invalidate_runtime_caches()
            return

        encoder = self._load_encoder(model_name)

        self.stdout.write(
            self.style.NOTICE(
                f"{total} chunk vektörleştiriliyor — model={model_name!r}, "
                f"batch_size={batch_size}, write_batch_size={write_batch_size}."
            )
        )

        t0 = time.perf_counter()
        processed = 0
        pending: list[ChunkEmbedding] = []

        for chunk_batch in self._iter_in_batches(chunks_qs, batch_size):
            texts = [c.chunk_text for c in chunk_batch]
            vectors = encoder.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype(np.float32)

            for chunk_obj, vec in zip(chunk_batch, vectors):
                pending.append(
                    ChunkEmbedding(
                        chunk=chunk_obj,
                        vector=vec.tolist(),
                        embedding_model=model_name,
                        embedding_dim=int(vec.shape[0]),
                        chunk_hash=chunk_obj.chunk_hash,
                    )
                )

            if len(pending) >= write_batch_size:
                ChunkEmbedding.objects.bulk_create(pending, batch_size=write_batch_size)
                processed += len(pending)
                pending.clear()
                self.stdout.write(f"İlerleme: {processed}/{total} tamamlandı.")

        if pending:
            ChunkEmbedding.objects.bulk_create(pending, batch_size=write_batch_size)
            processed += len(pending)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._invalidate_runtime_caches()

        # Surface the encoded dimension so an operator can verify it matches
        # the runtime expectation (``paraphrase-multilingual-MiniLM-L12-v2``
        # is 384-d, same as ``all-MiniLM-L6-v2``).
        sample_dim = ChunkEmbedding.objects.first().embedding_dim if processed else 0
        self.stdout.write(
            self.style.SUCCESS(
                f"İşlem bitti: {processed} vektör yazıldı, "
                f"dim={sample_dim}, elapsed={elapsed_ms:.0f}ms."
            )
        )

    def _select_chunks_to_process(self, model_name: str):
        """Pick chunks that still need an embedding for *this* model.

        After ``--force`` the table is empty, so every chunk is fair game.
        Without ``--force`` we exclude only chunks that already have an
        embedding under the same model name; chunks with a stale-model
        embedding are blocked earlier by ``_guard_against_model_mismatch``
        and never reach this method.
        """
        existing_for_model = ChunkEmbedding.objects.filter(
            embedding_model=model_name,
        ).values_list("chunk_id", flat=True)
        return (
            PageChunk.objects.exclude(id__in=existing_for_model)
            .order_by("id")
            .only("id", "chunk_text", "chunk_hash")
        )

    def _guard_against_model_mismatch(self, target_model: str) -> None:
        """Refuse to mix vectors from different models in the same table."""
        stale = (
            ChunkEmbedding.objects.exclude(embedding_model=target_model)
            .values_list("embedding_model", flat=True)
            .distinct()
        )
        stale_models = list(stale[:5])
        if not stale_models:
            return
        raise CommandError(
            "Veritabanında farklı modelle üretilmiş vektörler var: "
            f"{stale_models!r}. Yeni hedef model {target_model!r}. "
            "Yeni embedding boyutu/uzayı uyuşmaz; farklı modelleri tek "
            "tabloda karıştıramayız.\n"
            "Çözüm:\n"
            "  python manage.py rebuild_embeddings\n"
            "veya\n"
            f"  python manage.py create_embeddings --force --model-name {target_model}"
        )

    @staticmethod
    def _iter_in_batches(qs, batch_size: int) -> Iterable[list[PageChunk]]:
        buf: list[PageChunk] = []
        for obj in qs.iterator(chunk_size=batch_size * 4):
            buf.append(obj)
            if len(buf) >= batch_size:
                yield buf
                buf = []
        if buf:
            yield buf

    def _load_encoder(self, model_name: str):
        # Imported lazily so ``--help`` is fast and missing torch only
        # bites at execution time.
        from sentence_transformers import SentenceTransformer

        self.stdout.write(f"SentenceTransformer yükleniyor: {model_name!r} ...")
        return SentenceTransformer(model_name)

    @staticmethod
    def _invalidate_runtime_caches() -> None:
        """Drop in-process LRUs so the next ``/ask`` rebuilds the matrix.

        The retrieval helpers in :mod:`chatbot.services.embedding` cache
        both the embedding matrix (per source filter) and the loaded
        SentenceTransformer instance. If we left a stale matrix in place
        the very next request would dot a query vector from the new model
        against vectors from the old run.
        """
        try:
            from chatbot.services import embedding as svc

            svc._embedding_matrix_pack.cache_clear()
            svc._sentence_transformer_for_model.cache_clear()
            logger.info("Runtime embedding caches invalidated.")
        except Exception as exc:  # pragma: no cover — best-effort cleanup
            logger.warning("Cache invalidation skipped (%s)", exc)

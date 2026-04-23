from __future__ import annotations
from typing import Any
from django.core.management.base import BaseCommand, CommandError
from chatbot.models import ChunkEmbedding, PageChunk
from rag.document_loader import EXPECTED_EMBEDDING_MODEL
from rag.embedding_store import build_faiss_index

class Command(BaseCommand):
    help = "PageChunk verilerini vektörize eder ve ChunkEmbedding tablosuna kaydeder."

    def add_arguments(self, parser):
        parser.add_argument("--batch-size", type=int, default=64, help="Yapay zekanın tek seferde işleyeceği parça sayısı")
        parser.add_argument("--write-batch-size", type=int, default=100, help="Veritabanına toplu yazma boyutu")
        parser.add_argument("--force", action="store_true", help="Varsa eski embeddingleri silip baştan oluştur.")

    def handle(self, *args, **options):
        batch_size = options["batch_size"]
        write_batch_size = options["write_batch_size"]
        force = options["force"]
        model_name = EXPECTED_EMBEDDING_MODEL

        if force:
            self.stdout.write(self.style.WARNING("Eski embeddingler temizleniyor..."))
            ChunkEmbedding.objects.all().delete()

        pagechunk_total = PageChunk.objects.count()
        embedding_total = ChunkEmbedding.objects.count()

        # Sadece embedding'i olmayan chunk'ları al
        existing_chunk_ids = ChunkEmbedding.objects.values_list('chunk_id', flat=True)
        chunks_to_process = PageChunk.objects.exclude(id__in=existing_chunk_ids).order_by('id')
        
        total = chunks_to_process.count()
        if total == 0:
            self.stdout.write(
                self.style.SUCCESS(
                    "İşlenecek yeni veri bulunamadı. "
                    f"(PageChunk toplam={pagechunk_total}, ChunkEmbedding toplam={embedding_total})"
                )
            )
            if pagechunk_total == 0:
                self.stdout.write(
                    self.style.WARNING(
                        "PageChunk tablosu boş. Önce kaynak veriyi içeri aktarın:\n"
                        "  - Lokal .txt dosyaları: python manage.py ingest_txt_data --data-dir ./data\n"
                        "  - Web crawl: python manage.py ingest_acibadem --max-pages 150\n"
                        "Sonra tekrar çalıştırın: python manage.py create_embeddings"
                    )
                )
            return

        self.stdout.write(self.style.NOTICE(f"{total} parça veri vektörleştiriliyor..."))

        pending_instances = []
        processed_count = 0

        # Gruplar halinde işle (Hafıza dostu)
        for i in range(0, total, batch_size):
            batch = chunks_to_process[i:i + batch_size]
            texts = [c.chunk_text for c in batch]
            
            # Projenin kendi FAISS/Embedding mantığını kullan
            store = build_faiss_index(chunks=texts, embedding_model_name=model_name)
            
            for idx, chunk in enumerate(batch):
                vector = store.index.reconstruct(idx).tolist()
                pending_instances.append(ChunkEmbedding(
                    chunk=chunk,
                    vector=vector,
                    embedding_model=model_name,
                    embedding_dim=len(vector),
                    chunk_hash=chunk.chunk_hash
                ))
            
            # Toplu Yazma (Bulk Create)
            if len(pending_instances) >= write_batch_size:
                ChunkEmbedding.objects.bulk_create(pending_instances)
                processed_count += len(pending_instances)
                pending_instances = []
                self.stdout.write(f"İlerleme: {processed_count}/{total} tamamlandı.")

        # Kalanları yaz
        if pending_instances:
            ChunkEmbedding.objects.bulk_create(pending_instances)
            processed_count += len(pending_instances)

        self.stdout.write(self.style.SUCCESS(f"İşlem bitti! {processed_count} adet vektör oluşturuldu."))
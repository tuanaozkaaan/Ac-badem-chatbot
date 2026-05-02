# Acibadem Üniversitesi — Yerel RAG Sohbet Asistanı (AcuRate)

Django tabanlı, **Retrieval-Augmented Generation (RAG)** mimarisiyle çalışan; üniversiteye ait toplanmış metinlerden bağlam üreterek **Ollama** altında çalışan **Gemma** modeliyle yanıt veren bir sohbet prototipi. PostgreSQL’de saklanan gömütlü vektörler (FAISS + **Sentence Transformers**) ile tarayıcı oturumuna bağlı konuşma geçmişi birleştirilmiştir.

Bu depo, ders / jüri sunumunda “uçtan uca çalışan, güvenlik ve işletim açısından olgunlaştırılmış bir referans uygulama” olarak değerlendirilebilir.

---

## Mimari Özet

| Katman | Teknoloji | Not |
|--------|-----------|-----|
| Web / API | Django 5, Gunicorn (prod), WhiteNoise | HTTP katmanı ince; iş mantığı `chatbot/services` altında |
| Veri | PostgreSQL | Sayfa, parça, gömü ve konuşma kayıtları |
| RAG | FAISS, sentence-transformers | Sorgu gömümü ve benzerlik araması |
| LLM | Ollama (`OLLAMA_MODEL`, örn. `gemma:7b`) | Üretim parametreleri ve zaman aşımı yapılandırılabilir |
| Ön yüz | Şablon + statik CSS/JS modülleri | CSRF korumalı `fetch`, oturuma bağlı geçmiş |

**Kritik tasarım ilkesi:** `chatbot/services` HTTP’ten bağımsızdır; böylece birim testleri ve ileride DRF ViewSet’lere geçiş sadeleşir.

---

## Güvenlik ve Uyumluluk (Özet Epikriz)

- **Fail-closed üretim:** `DEBUG=0` iken zayıf `SECRET_KEY`, joker `ALLOWED_HOSTS` veya boş CORS listesi uygulamayı başlatmaz (`ImproperlyConfigured`).
- **CSRF:** `/ask` POST çağrıları için `ensure_csrf_cookie` + istemci tarafında `X-CSRFToken`.
- **IDOR önleme:** `Conversation.session_key` ile sohbet sahipliği; yetkisiz erişimde **404** (varlık sızdırmamak için bilinçli olarak 403 yerine).
- **Ağ:** Üretim `docker-compose.yml` içinde `db` ve `ollama` varsayılan olarak host’a port açmaz; geliştirme için `docker-compose.override.yml` kullanılır.

---

## Depo Yapısı (Seçilmiş)

```text
acu_chatbot/          Django proje ayarları (settings_test.py = yalnızca pytest)
chatbot/
  api/v1/             HTTP uçları, izinler, serileştirme
  services/           RAG, LLM, niyet, gömü, orchestrator
  management/commands/
rag/                  Belge işleme ve gömü hattı yardımcıları
static/chatbot/       Ayrıştırılmış CSS/JS
templates/            index iskeleti
docker/               entrypoint, Compose ile uyumlu başlatma
tests/                Pytest: smoke + güvenlik senaryoları
.github/workflows/  CI (PostgreSQL servis + pytest)
```

---

## Hızlı Başlangıç (Docker)

1. `.env.example` dosyasını `.env` olarak kopyalayın ve **REQUIRED** alanları doldurun.
2. Geliştirme deneyimi için override dosyasının varlığını doğrulayın (`docker-compose.override.yml`).
3. Ayağa kaldırma:

```bash
docker compose up --build
```

Web konteyneri migrasyon ve (yumuşak hata ile) `ollama_pull` çalıştırır; üretim modunda Gunicorn, geliştirmede `runserver` kullanılır (`DJANGO_DEBUG`).

**Veri kalıcılığı:** `docker compose down -v` kalıcı veri hacmini siler; dokümantasyon ve ekip içi uyarılarda bu komuttan kaçının.

---

## Yerel Geliştirme (Docker dışı)

```bash
python -m venv .venv
.\.venv\Scripts\activate   # Windows
pip install -r requirements-dev.txt
```

PostgreSQL ayarlarını `.env` üzerinden verin (`POSTGRES_HOST`, `POSTGRES_PORT`, …). Ardından:

```bash
python manage.py migrate
python manage.py create_embeddings   # RAG için gerekli
```

---

## Testler (Pytest)

Üretim şemasında **PostgreSQL `ArrayField`** kullanıldığı için test ayarı (`acu_chatbot.settings_test`) SQLite yerine Postgres’e bağlanır.

**Tek seferlik test veritabanı** (örnek; kullanıcı adınızı kullanın):

```bash
docker compose exec db psql -U acibadem -d acibadem_db -c "CREATE DATABASE acibadem_test OWNER acibadem;"
```

Çalıştırma (Windows PowerShell örneği):

```powershell
$env:POSTGRES_HOST = "127.0.0.1"
$env:POSTGRES_PORT = "5433"
$env:POSTGRES_USER = "acibadem"
$env:POSTGRES_PASSWORD = "…"
$env:POSTGRES_DB = "acibadem_db"
$env:POSTGRES_TEST_DB = "acibadem_test"
pytest tests/ -q
```

Senaryolar: **smoke** (`/health`, arayüz GET, konuşma listesi) ve **security** (CSRF reddi, IDOR’da 404, üretimde zayıf secret reddi).

---

## CI / CD (GitHub Actions)

`.github/workflows/main.yml` görevleri:

- PostgreSQL 16 servis konteyneri
- `pip install -r requirements-dev.txt`
- `pytest tests/`

Dal adınız `main` dışındaysa workflow’daki `branches` listesini güncelleyin.

---

## Bağımlılık Sabitleme Stratejisi

**Amacımız:** “Yarın bir transitif güncelleme geldi ve sistem davranışı sessizce değişti” riskini azaltmak.

1. **Doğrudan bağımlılıklar** `requirements.txt` içinde `==` ile sabitlenir (mevcut üretim kurulumu ile uyumlu sürümler).
2. **Geliştirici / CI araçları** `requirements-dev.txt` içinde tutulur (`-r requirements.txt` + pytest stack).
3. İleride **tam ünite replay** için aynı işletim sistemi üzerinde şu çıktıyı arşivlemek mümkündür:

```bash
pip install -r requirements-dev.txt
pip freeze > requirements.freeze.txt
```

> **Not:** `pip freeze` çıktısı tekerlek (wheel) ve platforma bağlıdır; Linux üretim imajı için donmuş listeyi tercihen Linux ortamında üretin.

Dockerfile, PyTorch’u CPU indeksinden ayrı kurmaya devam eder; `requirements.txt` içindeki `torch`/`sentence-transformers` çift kaynak çakışmasını önlemek için imaj içi kurulum sırasına bakınız.

---

## Veri Toplama ve RAG Üretimi

Kamu sayfalarına saygılı tarama ve OBS desteği için:

```bash
python manage.py ingest_acibadem --max-pages 150 --min-delay 1 --max-delay 2
```

Gömüleri güncellemek için `create_embeddings` komutunu kullanın.

---

## Eski / Alternatif Giriş Noktaları

Depoda tarihsel olarak `main.py`, `backend/` FastAPI prototipi vb. bulunabilir; canlı ürün yolu **Django + `chatbot/api/v1`** kabul edilir.

---

## Katkı ve Ekip İşbölümü

- `docker-compose.yml` üretim odaklı; `docker-compose.override.yml` yerel bind-mount ve portları taşır — dal çatışmalarını azaltmak için kasıtlı ayrım.
- Statik ön yüz dosyaları modülerleştirilmiştir (`static/chatbot/js/*.js`).

---

## Lisans ve Sorumluluk

Proje eğitim ve araştırma amaçlıdır; üretimde kullanımdan önce güvenlik gözden geçirmesi, yedekleme ve gözlemlenebilirlik (log/metrics) eklenmelidir.

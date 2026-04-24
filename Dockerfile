# Temel imaj
FROM python:3.11-slim-bookworm

# Çevresel değişkenler
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DJANGO_SETTINGS_MODULE=acu_chatbot.settings

WORKDIR /app

# --- KRİTİK ADIM: AI KÜTÜPHANELERİ İÇİN SİSTEM ARAÇLARI ---
# Bu satır eksik olduğu için derleme hatası alıyorsun
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılıkları kopyala
COPY requirements.txt /app/requirements.txt

# Zaman aşımını artırarak kur
RUN pip install --upgrade pip && \
    pip install --default-timeout=1000 --no-cache-dir -r /app/requirements.txt

# Playwright Chromium + sistem bağımlılıkları (slim imajda --with-deps gerekli)
RUN python -m playwright install --with-deps chromium

# Proje dosyalarını kopyala
COPY . /app
RUN chmod +x /app/docker/entrypoint.sh

EXPOSE 8000
ENTRYPOINT ["/app/docker/entrypoint.sh"]
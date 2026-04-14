#!/bin/sh

# Veritabanının hazır olmasını bekle
echo "Waiting for postgres..."
sleep 2

# Tabloları oluştur
python manage.py migrate --noinput

# Uygulamayı başlat
python manage.py runserver 0.0.0.0:8000
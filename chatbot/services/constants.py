"""Static text fragments used across orchestrator, prompts, and fallback responses.

Names keep their leading underscore to preserve the original module-private contract
while the strangler-fig refactor is in flight; downstream callers import them
explicitly from this module.
"""
from __future__ import annotations

# Crawled chunks often omit the postal block; official page:
# https://acibadem.edu.tr/kayit/iletisim/ulasim
ACIBADEM_GENERAL_FOCUS_BLOCK = (
    "[Genel tanıtım — odak]\n"
    "Acıbadem Üniversitesi kökleri itibarıyla sağlık bilimleri, tıp, hemşirelik, eczacılık ve benzeri alanlarda "
    "güçlü bir vakıf üniversitesidir; mühendislik ve diğer fakülteler program yelpazesinin parçasıdır. "
    "Genel 'üniversite nedir / kısa bilgi' sorularında tek bir mühendislik bölümünü veya tek kişiyi "
    "üniversitenin ana kimliği gibi sunma; önce sağlık ve çok disiplinli yapıyı özetle.\n"
)

OFFICIAL_CAMPUS_ADDRESS_BLOCK = (
    "[Resmî kampüs adresi ve iletişim — Acıbadem Üniversitesi; "
    "kaynak: acibadem.edu.tr/kayit/iletisim/ulasim]\n"
    "Kerem Aydınlar Kampüsü, Kayışdağı Cad. No:32, 34752 Ataşehir/İstanbul\n"
    "Telefon: 0216 500 44 44, 0216 576 50 76\n"
    "E-posta: info@acibadem.edu.tr\n"
)

_SAFE_FALLBACK_TR = (
    "Bu bilgi yerel veri kaynaklarında net olarak bulunamadı. "
    "En doğru ve güncel bilgi için Acıbadem Üniversitesi’nin resmi web sitesini kontrol etmeniz önerilir."
)
_SAFE_FALLBACK_EN = (
    "This information was not clearly found in the local data sources. "
    "For the most accurate and up-to-date information, please check Acıbadem University’s official website."
)

__all__ = [
    "ACIBADEM_GENERAL_FOCUS_BLOCK",
    "OFFICIAL_CAMPUS_ADDRESS_BLOCK",
    "_SAFE_FALLBACK_TR",
    "_SAFE_FALLBACK_EN",
]

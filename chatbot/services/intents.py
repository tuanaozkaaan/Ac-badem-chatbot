"""Intent detectors and the small set of canonical replies that ride alongside them.

Most functions here take a raw question string and return a boolean (intent match) or a
short string (canonical reply / context block). Two functions read text files from the
repo's ``data/`` directory; that disk I/O may move into a dedicated ``data_assets``
module during the F8 cleanup pass.

Dependencies are deliberately kept narrow:
    intents → services.constants, services.language
"""
from __future__ import annotations

from pathlib import Path

from chatbot.services.constants import (
    _CE_OVERVIEW_FALLBACK,
    _ENGINEERING_DEPARTMENTS_FALLBACK,
    _ENGINEERING_DEPARTMENTS_FILE,
)
from chatbot.services.language import _ascii_fold_turkish

# Resolves to <repo_root>/data regardless of which module imports us, because
# this file lives at <repo_root>/chatbot/services/intents.py.
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _ce_overview_context_block() -> str:
    """Short CE overview from repo data/ (Docker volume); fallback if file missing."""
    p = _DATA_DIR / "bilgisayar_muhendisligi.txt"
    try:
        if p.is_file():
            raw = p.read_text(encoding="utf-8", errors="replace").strip()
            if raw:
                return f"[Özet kaynak — Bilgisayar Mühendisliği lisans; resmî müfredat değildir]\n{raw}"
    except OSError:
        pass
    return _CE_OVERVIEW_FALLBACK


def _engineering_faculty_departments_intent(question: str) -> bool:
    q = _ascii_fold_turkish(question or "")
    has_faculty = "muhendislik ve doga bilimleri fakultesi" in q
    asks_for_list = any(
        n in q
        for n in (
            "hangi bolum",
            "bolumleri",
            "bolumler",
            "icerir",
            "nelerdir",
        )
    )
    return has_faculty and asks_for_list


def _engineering_faculty_departments_reply() -> str:
    p = _DATA_DIR / _ENGINEERING_DEPARTMENTS_FILE
    try:
        if p.is_file():
            lines = []
            for raw in p.read_text(encoding="utf-8", errors="replace").splitlines():
                line = raw.strip()
                if line.startswith("- "):
                    item = line[2:].strip()
                    if item:
                        lines.append(item)
            if lines:
                unique: list[str] = []
                seen: set[str] = set()
                for item in lines:
                    key = item.casefold()
                    if key in seen:
                        continue
                    seen.add(key)
                    unique.append(item)
                body = "\n".join(f"- {name}" for name in unique)
                return (
                    "Mühendislik ve Doğa Bilimleri Fakültesi şu bölümleri içerir:\n"
                    f"{body}\n\n"
                    "Not: Bu bilgi yerel veri dosyalarından derlenmiştir."
                )
    except OSError:
        pass
    return _ENGINEERING_DEPARTMENTS_FALLBACK


def _green_or_sustainable_campus_question(question: str) -> bool:
    """
    Sustainability / green campus topics. These must NOT trigger postal-address
    routing (word 'kampüs' alone would otherwise match address_intent).
    """
    raw = (question or "").lower()
    if any(
        x in raw
        for x in (
            "sürdürülebilir",
            "surdürülebilir",
            "yeşil kampüs",
            "yesil kampus",
            "çevre dostu",
            "iklim dostu",
            "karbon ayak",
            "karbon nötr",
            "leed",
            "eko kampüs",
            "eko kampus",
        )
    ):
        return True
    q = _ascii_fold_turkish(question or "")
    return any(
        x in q
        for x in (
            "surduurulebilir",
            "surdurulebilir",
            "sustainable",
            "green campus",
            "carbon neutral",
            "iklim",
            "leed",
            "eko kampus",
        )
    )


def _wants_postal_address_detail(q_lower: str) -> bool:
    if any(x in q_lower for x in ("address", "postal", "zip code", "zipcode")):
        return True
    if "adres" in q_lower:
        return True
    return any(
        p in q_lower
        for p in (
            "tam adres",
            "posta kodu",
            "cadde",
            "sokak",
            "detayli adres",
            "detaylı adres",
            "tam konum",
            "full address",
        )
    )


def _canonical_campus_address_reply(is_tr: bool) -> str:
    if is_tr:
        return (
            "Acıbadem Üniversitesi’nin posta tarzı kampüs adresi şöyledir:\n\n"
            "Kerem Aydınlar Kampüsü, Kayışdağı Cad. No:32, 34752 Ataşehir/İstanbul\n\n"
            "Telefon: 0216 500 44 44, 0216 576 50 76\n"
            "E-posta: info@acibadem.edu.tr"
        )
    return (
        "The official postal-style campus address is:\n\n"
        "Kerem Aydınlar Campus, Kayışdağı Avenue No:32, 34752 Ataşehir, Istanbul, Turkiye\n\n"
        "Phone: +90 216 500 44 44, +90 216 576 50 76\n"
        "Email: info@acibadem.edu.tr"
    )


def _detect_specific_faculty_focus(question: str) -> str | None:
    q = _ascii_fold_turkish(question or "")
    if "saglik bilimleri fakultesi" in q or "sağlık bilimleri fakültesi" in (question or "").lower():
        return "saglik_bilimleri"
    if "muhendislik ve doga bilimleri" in q or "mühendislik ve doğa bilimleri" in (question or "").lower():
        return "muhendislik_doga"
    return None


def _extract_faculty_phrase(question: str) -> str | None:
    q = _ascii_fold_turkish(question or "")
    known = (
        "saglik bilimleri fakultesi",
        "muhendislik ve doga bilimleri fakultesi",
        "tip fakultesi",
        "eczacilik fakultesi",
        "insan ve toplum bilimleri fakultesi",
    )
    for k in known:
        if k in q:
            return k
    return None


def _is_extractive_question(question: str) -> bool:
    q = _ascii_fold_turkish(question or "")
    cues = (
        "hangi bolumleri icerir",
        "hangi bolumler var",
        "hangi fakultede yer alir",
        "kimdir",
        "nerede",
        "iletisim bilgileri",
        "adres nedir",
    )
    return any(c in q for c in cues)


def _cs_engineering_lisans_intent(question: str) -> bool:
    """
    User asks about Computer *Engineering* (lisans), not the separate associate
    'Bilgisayar Programcılığı' (önlisans) program.
    """
    q = _ascii_fold_turkish(question or "")
    if "computer engineering" in q:
        return True
    if "bilgisayar" not in q:
        return False
    if "programcilik" in q and "muhendislik" not in q and "muhendisligi" not in q:
        return False
    return "muhendisligi" in q or "muhendislik" in q


def _cs_engineering_course_catalog_intent(question: str) -> bool:
    """
    Kullanıcı somut ders listesi / sınıf / müfredat / kod istiyor; genel 'alan tanımı' değil.
    Bu durumda repo özet dosyası (ce_block) enjekte edilmemeli — yalnızca DB/OBS chunk'ları.
    """
    if not _cs_engineering_lisans_intent(question):
        return False
    q = _ascii_fold_turkish(question or "")
    needles = (
        "1. sinif",
        "2. sinif",
        "3. sinif",
        "4. sinif",
        "birinci sinif",
        "ikinci sinif",
        "ucuncu sinif",
        "dorduncu sinif",
        "sinif ders",
        "siniftaki ders",
        "ders list",
        "derslerin",
        "dersleri",
        "dersler",
        "hangi ders",
        "hangi dersler",
        "dersler var",
        "programinda hangi",
        "programda hangi",
        "mufredat",
        "müfredat",
        "katalog",
        "curriculum",
        "syllabus",
        "ders kodu",
        "ders kod",
        "kredi",
        "akts",
        "yariyil",
        "yarıyıl",
        "donem",
        "dönem",
        "bologna",
        "program ciktisi",
        "program çıktısı",
        "ogrenme ciktisi",
        "öğrenme çıktısı",
    )
    return any(n in q for n in needles)


def _asks_subunits_of_named_faculty(question: str) -> bool:
    """
    Tek bir fakülte/yüksekokul adı verilmiş ve altındaki bölüm/program soruluyorsa True.
    Bu durumda _faculty_department_catalog_intent açılmamalı — yoksa tüm üniversite OR araması
    ve ağır dept_catalog yolu tetiklenir (dakikalarca sürebilir, bağlam da dağılır).
    """
    qf = _ascii_fold_turkish(question or "")
    if "fakultes" not in qf:
        return False
    if not any(
        x in qf
        for x in (
            "hangi bolum",
            "hangi program",
            "nelerdir",
            "icerir",
            "hangi lisans",
            "hangi onlisans",
        )
    ):
        return False
    units = (
        "muhendislik ve doga",
        "tip fakulte",
        "eczacilik fakulte",
        "saglik bilimleri fakulte",
        "guzel sanatlar",
        "hukuk fakulte",
        "iletisim fakulte",
        "dis hekimligi",
        "egitim bilimleri fakulte",
        "fen edebiyat",
        "yabanci diller",
    )
    return any(u in qf for u in units)


def _general_acibadem_intro_intent(question: str) -> bool:
    """
    'Üniversite hakkında kısa bilgi' gibi genel tanıtım — BM/tek programa kilitlenmesin diye retrieval + prompt ayarı.
    """
    qf = _ascii_fold_turkish(question or "")
    if "acibadem" not in qf and "aci badem" not in qf:
        return False
    if _cs_engineering_lisans_intent(question) or _cs_engineering_course_catalog_intent(question):
        return False
    if "bilgisayar" in qf or "computer" in qf or "muhendisligi" in qf:
        return False
    hints = (
        "hakkinda",
        "bilgi ver",
        "kisaca",
        "kisa bilgi",
        "nedir",
        "tanit",
        "tanitir misin",
        "genel bilgi",
        "universiteyi anlat",
        "universite hakkinda",
        "universitesi hakkinda",
        "hakkinda bilgi",
    )
    return any(h in qf for h in hints)


def _faculty_department_catalog_intent(question: str) -> bool:
    """Broad questions asking which faculties/departments/programs exist at the university."""
    ql = (question or "").lower()
    qf = _ascii_fold_turkish(question or "")
    if _asks_subunits_of_named_faculty(question):
        return False
    needles = (
        "hangi bölüm",
        "hangi bolum",
        "bölümler",
        "bolumler",
        "tüm bölüm",
        "tum bolum",
        "hangi fakülte",
        "hangi fakulte",
        "fakülteler",
        "fakulteler",
        "hangi program",
        "hangi programlar",
        "bölüm var",
        "bolum var",
        "fakülte var",
        "fakulte var",
    )
    if any(n in ql for n in needles):
        return True
    if any(x in ql for x in ("departments", "faculties", "schools at")):
        return True
    if "hangi" in qf and ("bolum" in qf or "fakulte" in qf or "program" in qf):
        return True
    # "Eczacılık fakültesini anlat" gibi tek fakülte sayfası — eskisi gibi geniş bağlam + çoklu URL.
    if "fakultes" in qf and any(
        x in qf for x in ("anlat", "tanit", "acikla", "nedir", "hakkinda", "bilgi ver")
    ):
        return True
    return False


__all__ = [
    "_ce_overview_context_block",
    "_engineering_faculty_departments_intent",
    "_engineering_faculty_departments_reply",
    "_green_or_sustainable_campus_question",
    "_wants_postal_address_detail",
    "_canonical_campus_address_reply",
    "_detect_specific_faculty_focus",
    "_extract_faculty_phrase",
    "_is_extractive_question",
    "_cs_engineering_lisans_intent",
    "_cs_engineering_course_catalog_intent",
    "_asks_subunits_of_named_faculty",
    "_general_acibadem_intro_intent",
    "_faculty_department_catalog_intent",
]

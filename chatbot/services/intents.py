"""Intent detectors and the small set of canonical replies that ride alongside them.

Most functions here take a raw question string and return a boolean (intent match) or a
short string (canonical reply / context block). Department-specific intents (Computer
Engineering, etc.) were removed in Adım 5.0 because the centralized
:mod:`chatbot.services.query_parser` now extracts the same signal in a generic way
across every program in :data:`metadata_enricher.DEPARTMENT_SLUG_MAP`.

Dependencies are deliberately kept narrow:
    intents → services.language
"""
from __future__ import annotations

from chatbot.services.language import _ascii_fold_turkish


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
    'Üniversite hakkında kısa bilgi' gibi genel tanıtım — tek programa kilitlenmesin diye
    retrieval + prompt ayarı. Department-specific intent gating (formerly Computer Engineering)
    was removed in Adım 5.0; the parser now flags department questions explicitly.
    """
    qf = _ascii_fold_turkish(question or "")
    if "acibadem" not in qf and "aci badem" not in qf:
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
    "_green_or_sustainable_campus_question",
    "_wants_postal_address_detail",
    "_canonical_campus_address_reply",
    "_detect_specific_faculty_focus",
    "_extract_faculty_phrase",
    "_is_extractive_question",
    "_asks_subunits_of_named_faculty",
    "_general_acibadem_intro_intent",
    "_faculty_department_catalog_intent",
]

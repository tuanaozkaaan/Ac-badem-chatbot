"""Keyword/scoring-based retrieval over PageChunk plus the local /data/ fallback.

This module is intentionally embedding-free: vector search lives in
``chatbot.services.embedding``. The two retrieval strategies are parallel
capabilities; the orchestrator decides which (or both) to use per request.

Allowed dependency direction:
    retrieval → services.context_select, services.intents, services.language
"""
from __future__ import annotations

import heapq
import logging
import os
import re
from pathlib import Path

from django.db.models import Q

from chatbot.services.context_select import _faculty_richness
from chatbot.services.intents import (
    _asks_subunits_of_named_faculty,
    _cs_engineering_course_catalog_intent,
    _cs_engineering_lisans_intent,
    _faculty_department_catalog_intent,
    _general_acibadem_intro_intent,
    _green_or_sustainable_campus_question,
)
from chatbot.services.language import _ascii_fold_turkish, _extract_keywords

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def retrieve_context(question: str, k: int = 5) -> str:
    """
    Scoring-based retrieval over PageChunk using ONLY Python logic (no embeddings).
    Weights:
      - title: highest
      - section: medium
      - chunk_text: lowest
    Also applies intent-based boosts/penalties for admissions vs internship queries.

    Candidate chunks are scored (including negative/zero), then top *k* by score are returned.
    To avoid scanning the entire table on large ingests, candidates are narrowed with
    `icontains` OR filters on keywords (plus a short prefix for long tokens) and capped by
    `RETRIEVE_MAX_CANDIDATES` (default 1000). If no keyword matches, the most recently updated
    chunks up to that cap are used instead.
    """
    keywords = _extract_keywords(question)
    # Question words widen OR-matches without helping entity questions ("kimdir" on half the site).
    lookup_noise = frozenset(
        {"kimdir", "nedir", "midir", "nerede", "nasıl", "nasil", "hangi", "neden", "niçin", "nicin", "kim", "ne"}
    )
    keywords = [k for k in keywords if k not in lookup_noise]
    q_lower = (question or "").lower()
    cs_eng_intent = _cs_engineering_lisans_intent(question)
    course_catalog_intent = _cs_engineering_course_catalog_intent(question)
    green_campus_q = _green_or_sustainable_campus_question(question)
    dept_catalog_intent = _faculty_department_catalog_intent(question)
    general_intro_intent = _general_acibadem_intro_intent(question)

    # Multi-word phrases (not produced by whitespace tokenization) improve DB icontains narrowing.
    extra_lookup_terms: list[str] = []
    if "bilgisayar mühendisliği" in q_lower or "bilgisayar muhendisligi" in q_lower:
        extra_lookup_terms.extend(
            ["bilgisayar mühendisliği", "bilgisayar muhendisligi", "bilgisayar mühendis", "bilgisayar muhendis"]
        )
    if cs_eng_intent:
        extra_lookup_terms.extend(
            [
                "mühendislik fakültesi",
                "muhendislik fakultesi",
                "computer engineering",
                "lisans programı",
                "lisans programi",
            ]
        )
    if course_catalog_intent:
        extra_lookup_terms.extend(
            [
                "obs.acibadem",
                "obs.acibadem.edu.tr",
                "bologna",
                "müfredat",
                "mufredat",
                "ders bilgi",
                "ders kodu",
                "akts",
                "kredi",
                "yarıyıl",
                "yariyil",
                "güz",
                "guz",
                "bahar",
                "bilgisayar mühendisliği",
                "bilgisayar muhendisligi",
                "öğrenim",
                "ogrenim",
            ]
        )
    if green_campus_q:
        extra_lookup_terms.extend(
            [
                "sürdürülebilir",
                "surdurulebilir",
                "sustainable",
                "yeşil kampüs",
                "yesil kampus",
                "iklim",
                "çevre",
                "karbon",
            ]
        )
    if "bölüm başkanı" in q_lower or "bolum baskani" in q_lower:
        extra_lookup_terms.extend(["bölüm başkanı", "bolum baskani", "bölüm başkan", "bolum baskan"])
    if dept_catalog_intent:
        extra_lookup_terms.extend(
            [
                "fakülte",
                "fakulte",
                "fakülteler",
                "fakulteler",
                "mühendislik",
                "muhendislik",
                "tıp fakültesi",
                "tip fakultesi",
                "eczacılık",
                "hemşirelik",
                "beslenme",
                "fizyoterapi",
                "sağlık bilimleri",
                "saglik bilimleri",
                "mühendislik ve doğa",
                "muhendislik ve doga",
                "güzel sanatlar",
                "guzel sanatlar",
                "hukuk fakültesi",
                "hukuk fakultesi",
                "diş hekimliği",
                "dis hekimligi",
                "üniversitemiz",
                "universitemiz",
                "lisans",
                "önlisans",
                "onlisans",
            ]
        )
    if _asks_subunits_of_named_faculty(question):
        qfx = _ascii_fold_turkish(question or "")
        if "muhendislik ve doga" in qfx:
            extra_lookup_terms.extend(
                [
                    "mühendislik ve doğa bilimleri",
                    "muhendislik ve doga bilimleri",
                    "muhendislik ve doga",
                ]
            )
        if "tip fakulte" in qfx:
            extra_lookup_terms.extend(["tıp fakültesi", "tip fakultesi"])
        if "eczacilik fakulte" in qfx:
            extra_lookup_terms.extend(["eczacılık fakültesi", "eczacilik fakultesi"])
        if "saglik bilimleri fakulte" in qfx:
            extra_lookup_terms.extend(["sağlık bilimleri", "saglik bilimleri"])
    if general_intro_intent:
        extra_lookup_terms.extend(
            [
                "sağlık bilimleri",
                "saglik bilimleri",
                "tıp fakültesi",
                "tip fakultesi",
                "vakıf üniversitesi",
                "vakif universitesi",
                "Acıbadem Sağlık",
                "hemşirelik",
                "hemsirelik",
                "eczacılık",
                "kuruluş",
                "kurulus",
            ]
        )

    # Intent detection (priority matters):
    # - If question mentions staj/internship => internship intent
    # - Else if question mentions apply/admission terms => general admission intent
    internship_intent = any(t in q_lower for t in ("staj", "internship"))
    admission_intent = (not internship_intent) and any(
        t in q_lower for t in ("başvuru", "basvuru", "admission", "apply", "application")
    )
    address_intent = (not green_campus_q) and any(
        t in q_lower
        for t in (
            "adres",
            "address",
            "kampüs",
            "kampus",
            "campus",
            "konum",
            "location",
            "ulaşım",
            "ulasim",
            "nerede",
            "nasıl gid",
            "nasil gid",
        )
    )
    faculty_head_intent = any(
        t in q_lower
        for t in (
            "bölüm başkanı",
            "bolum baskani",
            "bölüm başkan",
            "bolum baskan",
            "program başkanı",
            "program baskani",
            "department head",
            "department chair",
            "müdür",
            "mudur",
        )
    )
    if address_intent:
        # Keep these specific (avoid broad "İstanbul" OR matches that flood candidates).
        extra_lookup_terms.extend(
            [
                "kerem aydınlar",
                "kerem aydinlar",
                "iletişim ve ulaşım",
                "iletisim ve ulasim",
                "atköy",
                "atkoy",
                "inönü",
                "inonu",
                "ulaşım",
                "ulasim",
            ]
        )

    admission_boost_terms = {
        "admission",
        "başvuru",
        "basvuru",
        "application",
        "apply",
        "kabul",
        "requirements",
        "öğrenci kabul",
        "ogrenci kabul",
    }
    admission_penalize_terms = {
        "staj",
        "internship",
        "staj başvurusu",
        "staj basvurusu",
        "career",
        "kariyer",
    }

    internship_boost_terms = {
        "staj",
        "internship",
        "staj başvurusu",
        "staj basvurusu",
        "staj yönergesi",
        "staj yonergesi",
        "staj defteri",
        "staj komisyonu",
    }
    internship_penalize_terms = {
        "admission",
        "başvuru",
        "basvuru",
        "application",
        "apply",
        "öğrenci kabul",
        "ogrenci kabul",
        "requirements",
        "kabul",
    }

    noisy_penalize_terms = {"duyuru", "haber", "etkinlik", "announcement", "news", "event"}
    address_boost_terms = {
        "adres",
        "address",
        "iletişim",
        "iletisim",
        "ulaşım",
        "ulasim",
        "kampüs",
        "kampus",
        "kampüsü",
        "kampusu",
        "campus",
        "yerleşke",
        "yerleske",
    }
    address_penalize_terms = {
        # Common in program pages and tends to match "adresine" (email address) rather than campus address.
        "referans",
        "transkript",
        "ales",
        "başvuru",
        "basvuru",
        "application",
    }

    def _count_hits(hay: str, needle: str) -> int:
        if not hay or not needle:
            return 0
        return hay.count(needle)

    # Uzun sayfa metinlerinde her satırda tüm gövdeyi skorlamak CPU'da onlarca sn sürebilir; eşleştirme için önek yeter.
    _score_body_limit = int(os.environ.get("RETRIEVE_CHUNK_SCORE_CHARS", "5500"))
    _score_body_limit = max(2800, min(_score_body_limit, 120_000))

    def _score_row(row) -> int:
        title = (row.title or "").lower()
        section = (row.section or "").lower()
        raw_body = row.chunk_text or ""
        if len(raw_body) > _score_body_limit:
            raw_body = raw_body[:_score_body_limit]
        text = raw_body.lower()
        url = (getattr(row, "url", None) or "").lower()

        score = 0

        # Keyword scoring: title highest, section medium, chunk_text lowest.
        for kw in keywords[:10]:
            score += 8 * _count_hits(title, kw)
            score += 4 * _count_hits(section, kw)
            score += 2 * _count_hits(text, kw)

        if admission_intent:
            for t in admission_boost_terms:
                if t in title:
                    score += 40
                if t in section:
                    score += 24
                if t in text:
                    score += 14

            for t in admission_penalize_terms:
                if t in title:
                    score -= 90
                if t in section:
                    score -= 60
                if t in text:
                    score -= 40

        elif internship_intent:
            for t in internship_boost_terms:
                if t in title:
                    score += 40
                if t in section:
                    score += 24
                if t in text:
                    score += 14

            for t in internship_penalize_terms:
                if t in title:
                    score -= 80
                if t in section:
                    score -= 55
                if t in text:
                    score -= 35

        for t in noisy_penalize_terms:
            if t in title:
                score -= 18
            if t in section:
                score -= 12
            if t in text:
                score -= 8

        if address_intent:
            for t in address_boost_terms:
                if t in title:
                    score += 36
                if t in section:
                    score += 22
                if t in text:
                    score += 10

            for t in address_penalize_terms:
                if t in title:
                    score -= 40
                if t in section:
                    score -= 24
                if t in text:
                    score -= 14

            # Heuristic: lots of emails often means we matched "adresine" (email address).
            email_hits = text.count("@")
            if email_hits >= 2 and not any(x in text for x in ("kampüs", "kampus", "campus", "ataşehir", "atasehir")):
                score -= 60

            # Strongly prefer actual contact/transport pages when asking for address/location.
            if any(p in url for p in ("/iletisim", "/contact", "/ulasim", "/adres", "/yerleske", "/kampus", "/kampus/")):
                score += 140
            if ("iletişim" in title) or ("iletisim" in title) or ("contact" in title):
                score += 140

            # Academic program pages are usually not about physical location.
            if "/akademik/" in url and not any(p in url for p in ("/iletisim", "/contact", "/ulasim", "/adres")):
                score -= 35

            geo_markers = (
                "ataşehir",
                "atasehir",
                "istanbul",
                "i̇stanbul",
                "caddesi",
                "cad.",
                "mahalle",
                "mah.",
                "posta kodu",
                "pk.",
                "inönü",
                "inonu",
                "atköy",
                "atkoy",
            )
            geo_hits = sum(1 for g in geo_markers if g in text)
            if geo_hits >= 2:
                score += 110
            elif geo_hits == 1:
                score += 50
            if "no:" in text or " no:" in text:
                score += 28
            # Down-rank snippets that only place an office on a floor without street/district clues.
            if "kat" in text and any(o in text for o in ("ofis", "office", "acumed", "büro", "biro")):
                if not any(s in text for s in ("ataşehir", "atasehir", "istanbul", "caddesi", "mahalle", "inönü", "inonu")):
                    score -= 95

        if faculty_head_intent:
            if any(x in text for x in ("başkan", "baskan", "müdür", "mudur", "chair", "head of")):
                score += 28
            if any(x in title for x in ("başkan", "baskan", "müdür", "mudur", "chair")):
                score += 55
            if any(x in section for x in ("başkan", "baskan", "müdür", "mudur")):
                score += 35
            if "/akademik/" in url:
                score += 18

        # When user names a specific engineering/CS department, avoid ranking unrelated "bölüm başkanı" pages.
        cs_dept_focus = faculty_head_intent and any(
            t in q_lower for t in ("bilgisayar", "yazılım", "yazilim", "computer", "mühendislik", "muhendislik")
        )
        if cs_dept_focus:
            if any(t in text for t in ("bilgisayar", "yazılım", "yazilim", "computer science", "computer")):
                score += 55
            if any(t in title for t in ("bilgisayar", "yazılım", "yazilim", "computer")):
                score += 70
            if any(t in url for t in ("bilgisayar", "yazilim", "computer", "mühendislik", "muhendislik")):
                score += 45
            # Do not treat associate "programcılığı" pages as the engineering department chair source.
            if "bilgisayar-programciligi" in url or "bilgisayar-programcılığı" in url:
                score -= 180
            if any(x in url for x in ("muhendislik", "mühendislik", "engineering")):
                score += 85
            if ("sağlık yönetimi" in text or "saglik yonetimi" in text) and "bilgisayar" not in text and "yazılım" not in text:
                score -= 95
            if ("onlisans" in url or "onlisans" in text) and "bilgisayar" not in text:
                score -= 35
            for bad in (
                "saglik-yonetimi",
                "saglik-hizmetleri",
                "radyoterapi",
                "yabanci-diller",
                "ingilizce-hazirlik",
            ):
                if bad in url and "bilgisayar-programciligi" not in url and "muhendislik" not in url:
                    score -= 140

        if cs_eng_intent:
            u = url.lower()
            if "bilgisayar-programciligi" in u or "bilgisayar-programcılığı" in u:
                score -= 280
            if "onlisans" in u or "onlisans" in text:
                score -= 220
            if ("programcılık" in text or "programcilik" in text) and "muhendislik" not in text and "mühendislik" not in text:
                score -= 120
            if any(x in u for x in ("muhendislik", "mühendislik", "engineering", "lisans")):
                score += 100
            if "bilgisayar mühendisliği" in text or "bilgisayar muhendisligi" in text:
                score += 120
            if "mühendislik fakültesi" in text or "muhendislik fakultesi" in text:
                score += 90

        if green_campus_q:
            blob = f"{text} {title} {section}".lower()
            if any(
                x in blob
                for x in (
                    "sürdürülebilir",
                    "surdurulebilir",
                    "sustainable",
                    "yeşil",
                    "yesil",
                    "iklim",
                    "karbon",
                    "leed",
                    "çevre",
                    "eko",
                )
            ):
                score += 110
            st = (section or "").lower()
            if st == "contact_address" and not any(
                x in blob for x in ("sürdürülebilir", "surdurulebilir", "sustainable", "iklim", "karbon", "leed")
            ):
                score -= 150

        if dept_catalog_intent:
            tl = (title or "").lower()
            sl = (section or "").lower()
            if "fakülte" in tl or "fakulte" in tl or "fakülte" in sl or "fakulte" in sl:
                score += 70
            if sum(1 for w in ("tıp", "eczacılık", "mühendislik", "hemşirelik", "beslenme", "fizyoterapi") if w in text) >= 2:
                score += 45
            if "/akademik/" in url and ("lisans" in url or "onlisans" in url or "yuksekokul" in url):
                score += 35
            uroot = url.rstrip("/").lower()
            if uroot.endswith("acibadem.edu.tr") and any(
                x in text for x in ("fakülte", "fakulte", "tıp", "mühendislik", "eczacılık", "hemşirelik")
            ):
                score += 115
            # Tam fakülte listesinde çok sayfa var; tek fakülte URL'sine aşırı boost tüm sırayı eczacılıkta topluyor.
            if "eczacilik" in url or "eczacılık" in url:
                score += 28
            blob_fac = _ascii_fold_turkish(f"{title} {section} {text}")
            fr = _faculty_richness(blob_fac)
            if fr >= 70:
                score += min(220, fr + 40)
            elif fr >= 42:
                score += 85
            elif fr >= 14:
                score += 28

        if general_intro_intent:
            blob_gi = _ascii_fold_turkish(f"{title} {section} {text} {url}")
            u = url.lower()
            if any(
                x in u
                for x in (
                    "bilgisayar-muhendisligi",
                    "bilgisayar_muhendisligi",
                    "computer-engineering",
                )
            ):
                score -= 280
            if "bilgisayar" in title and "muhendis" in _ascii_fold_turkish(title):
                score -= 160
            if any(
                x in blob_gi
                for x in (
                    "saglik bilimleri",
                    "tip fakulte",
                    "tip fakültesi",
                    "eczacilik",
                    "hemsirelik",
                    "dis hekimligi",
                    "vakif universitesi",
                    "kurulus",
                    "saglik grubu",
                    "hastane",
                )
            ):
                score += 120

        if course_catalog_intent:
            st = (getattr(row, "source_type", None) or "").lower()
            if st == "obs":
                score += 110
            if "obs.acibadem" in url:
                score += 130
            blob_ct = f"{text} {title}"
            for kw in (
                "müfredat",
                "mufredat",
                "bologna",
                "akts",
                "kredi",
                "yarıyıl",
                "yariyil",
                "güz",
                "guz",
                "bahar",
                "ders kodu",
                "ders kod",
                "öğrenim",
                "ogrenim",
                "program çıktısı",
                "program ciktisi",
            ):
                if kw in blob_ct:
                    score += 38
            if re.search(r"\b[a-zçğıöşü]{2,5}\s*\d{3}\b", blob_ct, re.I):
                score += 72

        return score

    # 1) Prefer DB chunks if available — score a bounded candidate set, then take top k by score.
    try:
        from chatbot.models import PageChunk

        if PageChunk.objects.exists():
            # 6000 satırı tamamen RAM'de skorlamak yavaş; varsayılanı düşük tut (env ile artırılabilir).
            max_candidates = max(1, int(os.environ.get("RETRIEVE_MAX_CANDIDATES", "1000")))
            base_qs = PageChunk.objects.only(
                "chunk_text", "title", "section", "url", "source_type", "updated_at"
            ).order_by("-updated_at")

            lookup_terms: list[str] = []
            seen_terms: set[str] = set()
            for kw in keywords[:12]:
                if len(kw) < 3:
                    continue
                if kw not in seen_terms:
                    seen_terms.add(kw)
                    lookup_terms.append(kw)
                # Longer root for DB icontains (avoid e.g. "başkanı"[:5] -> "başka" false matches).
                if len(kw) >= 7:
                    plen = min(12, max(6, len(kw) - 1))
                    root = kw[:plen]
                    if len(root) >= 5 and root != kw and root not in seen_terms:
                        seen_terms.add(root)
                        lookup_terms.append(root)
                if len(lookup_terms) >= 22:
                    break

            for t in extra_lookup_terms:
                tt = (t or "").strip()
                if len(tt) >= 3 and tt not in seen_terms:
                    seen_terms.add(tt)
                    lookup_terms.append(tt)

            q_filter = Q()
            if lookup_terms:
                for term in lookup_terms:
                    q_filter |= (
                        Q(chunk_text__icontains=term)
                        | Q(title__icontains=term)
                        | Q(section__icontains=term)
                        | Q(url__icontains=term)
                    )

            candidate_qs = base_qs
            if address_intent:
                # Prefer real contact / transport / district lines instead of random "İstanbul" mentions.
                addr_anchor = (
                    Q(url__icontains="iletisim")
                    | Q(url__icontains="ulasim")
                    | Q(url__icontains="contact")
                    | Q(url__icontains="/adres")
                    | Q(chunk_text__icontains="ataşehir")
                    | Q(chunk_text__icontains="atasehir")
                    | Q(chunk_text__icontains="inönü")
                    | Q(chunk_text__icontains="inonu")
                    | Q(chunk_text__icontains="kerem aydınlar")
                    | Q(chunk_text__icontains="kerem aydinlar")
                )
                addr_qs = base_qs.filter(addr_anchor)
                if addr_qs.exists():
                    if lookup_terms:
                        merged = addr_qs.filter(q_filter)
                        candidate_qs = merged if merged.exists() else addr_qs
                    else:
                        candidate_qs = addr_qs
                elif lookup_terms:
                    narrowed = base_qs.filter(q_filter)
                    candidate_qs = narrowed if narrowed.exists() else base_qs[:max_candidates]
                else:
                    candidate_qs = base_qs[:max_candidates]
            elif lookup_terms:
                narrowed = base_qs.filter(q_filter)
                candidate_qs = narrowed if narrowed.exists() else base_qs[:max_candidates]
            else:
                candidate_qs = base_qs[:max_candidates]

            candidate_qs = candidate_qs[:max_candidates]

            def _iter_scored_rows():
                for row in candidate_qs.iterator(chunk_size=400):
                    yield (_score_row(row), str(row.updated_at or ""), row)

            pool_size = max(
                1,
                min(
                    max_candidates,
                    max(500, k * 28) if dept_catalog_intent else max(280, k * 22),
                ),
            )
            scored_pool = heapq.nlargest(
                pool_size,
                _iter_scored_rows(),
                key=lambda t: (t[0], t[1]),
            )
            # Düşük veya negatif skorlarda bile en iyi k adayı kullan: erken return "" hem RAG'ı
            # keser hem de aşağıdaki data/*.txt yedeğine düşmeyi engeller (yorumla uyumlu).

            logger.info("retrieve_context(db): question=%r pool=%s (max_cand=%s)", question, len(scored_pool), max_candidates)
            for rank, (s, _u, row) in enumerate(scored_pool[:k], start=1):
                logger.info("retrieve_context(db): #%s score=%s title=%r", rank, s, row.title)

            if dept_catalog_intent:
                picked: list = []
                url_counts: dict[str, int] = {}
                seen_pk: set[int] = set()
                # Liste sorusunda farklı URL'lerden parça al — aynı sitede 2 parça yerine 14 ayrı kaynak.
                max_per_url = 1

                def _take_row(row) -> bool:
                    pk = int(row.pk)
                    if pk in seen_pk:
                        return False
                    urlv = ((row.url or "")[:220] or "?").lower()
                    if url_counts.get(urlv, 0) >= max_per_url:
                        return False
                    picked.append(row)
                    seen_pk.add(pk)
                    url_counts[urlv] = url_counts.get(urlv, 0) + 1
                    return True

                for _s, _u, row in scored_pool:
                    if len(picked) >= k:
                        break
                    _take_row(row)
                for _s, _u, row in scored_pool:
                    if len(picked) >= k:
                        break
                    if int(row.pk) in seen_pk:
                        continue
                    picked.append(row)
                    seen_pk.add(int(row.pk))
                # İlk eşleşen parça sık sık yalnızca 1–2 fakülte içerir; en zengin "genel liste" parçasını seç.
                overview_row = None
                overview_score = -1
                overview_tuple: tuple[int, str, object] | None = None
                for _s, _u, row in scored_pool[:280]:
                    uu = (row.url or "").rstrip("/").lower()
                    if not uu.endswith("acibadem.edu.tr"):
                        continue
                    tx = (row.chunk_text or "") + " " + (row.title or "")
                    fr = _faculty_richness(_ascii_fold_turkish(tx.lower()))
                    # ~3+ farklı birim ipucu (≈42+) olan parça genelde tam liste / akademik özet sayfasıdır.
                    if fr < 42:
                        continue
                    if fr > overview_score or (
                        fr == overview_score and overview_tuple is not None and _s > overview_tuple[0]
                    ):
                        overview_score = fr
                        overview_tuple = (_s, _u, row)
                if overview_tuple is not None:
                    overview_row = overview_tuple[2]
                if overview_row is not None and int(overview_row.pk) not in {int(r.pk) for r in picked}:
                    picked = [overview_row] + picked[: max(0, k - 1)]
                top_rows = picked[:k]
            else:
                top_rows = [row for _, _, row in scored_pool[:k]]
            blocks: list[str] = []
            for c in top_rows:
                meta = " | ".join(
                    [p for p in [c.title or "", c.section or "", c.source_type or "", c.url or ""] if p]
                )
                blocks.append(f"[{meta}]\n{c.chunk_text}" if meta else c.chunk_text)

            merged_db = "\n\n---\n\n".join(blocks).strip()
            if merged_db:
                return merged_db
    except Exception:
        logger.exception("retrieve_context: DB veya skorlama hatası (yedeğe düşülüyor)")

    # 2) Fallback: local /data/*.txt — include all scored lines, take top k even if low score.
    try:
        from rag.document_loader import load_text_documents
        from rag.text_splitter import split_into_chunks
    except Exception:
        logger.exception("retrieve_context: rag.* modules unavailable; returning empty context")
        return ""

    data_dir = str(_DATA_DIR)
    try:
        docs = load_text_documents(data_dir)
        chunks = split_into_chunks(docs, chunk_size=900, chunk_overlap=150)
    except Exception:
        logger.exception("retrieve_context: data/ fallback load failed; returning empty context")
        return ""

    scored_txt: list[tuple[int, int, str]] = []
    for idx, ch in enumerate(chunks):
        text_l = ch.lower()
        score = 0
        for kw in keywords[:10]:
            score += text_l.count(kw)
        if admission_intent:
            for t in admission_boost_terms:
                if t in text_l:
                    score += 4
            for t in admission_penalize_terms:
                if t in text_l:
                    score -= 10
        elif internship_intent:
            for t in internship_boost_terms:
                if t in text_l:
                    score += 4
            for t in internship_penalize_terms:
                if t in text_l:
                    score -= 10

        for t in noisy_penalize_terms:
            if t in text_l:
                score -= 2
        scored_txt.append((score, -idx, ch))

    scored_txt.sort(key=lambda x: (x[0], x[1]), reverse=True)
    top = [c for _, _, c in scored_txt[:k]]
    return "\n\n---\n\n".join(top).strip()


__all__ = ["retrieve_context"]

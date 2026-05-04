"""Regression tests for :mod:`chatbot.services.query_parser`.

Adım 5.3 fix
------------
The Adım 5.0 cleanup deleted the hard-coded Computer Engineering catalog
intent, leaving the parser with no way to flag plain "dersler" / "courses"
questions. The hybrid retriever then returned program-overview chunks
instead of course-tagged chunks, and the LLM produced the stock
"no info" reply. These tests pin the new pattern so the regression
cannot resurface silently.
"""
from __future__ import annotations

import pytest

from chatbot.ingestion.metadata_enricher import ContentType
from chatbot.services.query_parser import parse_query


# ---------------------------------------------------------------------------
# Adım 5.3 — bare-plural "dersler" / "courses" must tag BOLOGNA_COURSE
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "question, expected_dept",
    [
        # Turkish, present participle suffix.
        ("Bilgisayar mühendisliğinde olan dersler nedir?", "Bilgisayar Mühendisliği"),
        # Turkish, plain plural genitive.
        ("Bilgisayar mühendisliği dersleri nelerdir?", "Bilgisayar Mühendisliği"),
        # Turkish, accusative inflection.
        (
            "Endüstri mühendisliği derslerini listeler misin?",
            "Endüstri Mühendisliği",
        ),
        # English, bare plural.
        ("What courses does Computer Engineering have?", "Bilgisayar Mühendisliği"),
        # English, "course list" phrase.
        (
            "Can you share the course list for Industrial Engineering?",
            "Endüstri Mühendisliği",
        ),
    ],
)
def test_bare_plural_courses_tag_bologna_course(question: str, expected_dept: str) -> None:
    filters = parse_query(question)
    assert filters.department == expected_dept, (
        f"department parse regressed on {question!r}: got {filters.department!r}"
    )
    assert ContentType.BOLOGNA_COURSE in filters.content_types, (
        f"BOLOGNA_COURSE not flagged on {question!r}: got {filters.content_types!r}"
    )


# ---------------------------------------------------------------------------
# Existing patterns must still fire (regression guard)
# ---------------------------------------------------------------------------
def test_specific_course_code_still_works() -> None:
    """The CSE101-style hits the user already confirmed working."""
    filters = parse_query("CSE101 dersi kaç AKTS?")
    assert filters.course_code == "CSE101"
    assert ContentType.BOLOGNA_COURSE in filters.content_types


def test_curriculum_phrase_still_works() -> None:
    filters = parse_query("Bilgisayar mühendisliği müfredatı nedir?")
    assert filters.department == "Bilgisayar Mühendisliği"
    assert ContentType.BOLOGNA_COURSE in filters.content_types


def test_curriculum_english_still_works() -> None:
    filters = parse_query("What is the curriculum of Computer Engineering?")
    assert filters.department == "Bilgisayar Mühendisliği"
    assert ContentType.BOLOGNA_COURSE in filters.content_types


def test_ders_with_qualifier_still_works() -> None:
    """`ders` + qualifier ("ders kodu", "ders programı", ...) must keep firing."""
    for qualifier in ("ders kodu", "ders listesi", "ders kataloğu", "ders içeriği", "ders programı"):
        filters = parse_query(f"Bilgisayar mühendisliği {qualifier} nedir?")
        assert ContentType.BOLOGNA_COURSE in filters.content_types, (
            f"BOLOGNA_COURSE not flagged for qualifier {qualifier!r}"
        )


# ---------------------------------------------------------------------------
# Negative tests — guard against false positives introduced by the new pattern
# ---------------------------------------------------------------------------
def test_singular_ders_does_not_tag_course_catalog() -> None:
    """`ders` alone (singular) must NOT trigger BOLOGNA_COURSE — too generic.

    Without this guard we would tag conversational uses ("Bu ders güzeldi.")
    and pollute retrieval with unrelated curriculum chunks.
    """
    filters = parse_query("Bu ders güzeldi.")
    assert ContentType.BOLOGNA_COURSE not in filters.content_types


def test_courseware_does_not_tag_course_catalog() -> None:
    """English regex must use word boundaries so 'coursework' / 'concourse' don't match."""
    filters = parse_query("The university has innovative coursework systems.")
    assert ContentType.BOLOGNA_COURSE not in filters.content_types


def test_unrelated_question_returns_empty_filters() -> None:
    """Sanity: a question with neither department nor curriculum cues stays empty."""
    filters = parse_query("Adres nedir?")
    assert filters.department is None
    assert filters.course_code is None
    assert filters.content_types == ()


# ---------------------------------------------------------------------------
# matched_terms diagnostic should record the firing intent (debug aid)
# ---------------------------------------------------------------------------
def test_matched_terms_records_intent() -> None:
    filters = parse_query("Bilgisayar mühendisliği dersleri nelerdir?")
    joined = " | ".join(filters.matched_terms)
    assert "dept:Bilgisayar Mühendisliği" in joined
    assert "intent:bologna_course,bologna_program" in joined

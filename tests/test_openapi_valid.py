"""Lightweight CI guardrail for ``docs/openapi.yaml``.

The OpenAPI document is the published wire-format contract for the chatbot;
when ``chatbot/api/v1/serializers.py`` changes shape, the spec must follow
in the same patch. This test only enforces that the file is structurally
valid OpenAPI 3.x — semantic alignment with serializers is covered by
``test_ask_contract.py``.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_PATH = REPO_ROOT / "docs" / "openapi.yaml"


def _load_spec() -> dict:
    yaml = pytest.importorskip("yaml", reason="PyYAML is a dev dependency for spec validation")
    with SPEC_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_openapi_yaml_exists() -> None:
    assert SPEC_PATH.is_file(), f"missing spec at {SPEC_PATH}"


def test_openapi_yaml_validates() -> None:
    pytest.importorskip(
        "openapi_spec_validator",
        reason="openapi-spec-validator is a dev dependency (see requirements-dev.txt)",
    )
    from openapi_spec_validator import validate

    spec = _load_spec()
    # Raises on malformed spec; surface the exception verbatim so a CI failure
    # points at the offending JSON path.
    validate(spec)


def test_openapi_declares_ask_v1_contract() -> None:
    """Smoke-check: the v1 endpoints we actually ship are documented."""
    spec = _load_spec()
    paths = spec.get("paths") or {}
    assert "/api/v1/ask" in paths, "ask endpoint missing from spec"
    assert "/api/v1/health" in paths, "health endpoint missing from spec"
    assert "/api/v1/conversations/" in paths, "conversations list endpoint missing"
    assert "/api/v1/conversations/{id}/" in paths, "conversation detail endpoint missing"

    ask_post = paths["/api/v1/ask"].get("post") or {}
    success = (((ask_post.get("responses") or {}).get("200") or {}).get("content") or {})
    schema_ref = ((success.get("application/json") or {}).get("schema") or {}).get("$ref")
    assert schema_ref == "#/components/schemas/AskResponse", (
        f"AskResponse schema not wired to /api/v1/ask 200; got {schema_ref!r}"
    )

    components = (spec.get("components") or {}).get("schemas") or {}
    for name in ("AskRequest", "AskResponse", "RetrievedChunk", "QueryFilters",
                 "LatencyMs", "ErrorResponse"):
        assert name in components, f"required component schema missing: {name}"

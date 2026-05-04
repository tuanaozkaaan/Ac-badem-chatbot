"""End-to-end /ask pipeline.

This is the only services-layer module allowed to know that an /ask call has many
stages (intents → retrieval → LLM → post-processing → persistence). It stays
HTTP-agnostic: returns ``(payload_dict, status_code, AskMeta)`` for the HTTP
adapter to wrap into a JsonResponse. The ``payload_dict`` keeps the legacy
``{conversation_id, answer | detail}`` shape; ``AskMeta`` is the side-channel
the v1 view (Adım 5.1) uses to project retrieval/timing/source onto the wire
without forcing services to depend on serializers.

Allowed dependency direction (top of the DAG):
    ask_orchestrator → constants, language, intents, context_select, extractive,
                       embedding, query_parser, llm_client, prompts, conversation_repo

Retrieval (Adım 5.0.5)
----------------------
Production `/ask` uses :func:`chatbot.services.embedding._retrieve_top_chunks_by_embedding`
(parser + hybrid metadata filter + global cosine). Legacy keyword-only retrieval was
removed from this path.
"""
from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

from chatbot.ingestion.metadata_enricher import ContentType
from chatbot.services.constants import (
    ACIBADEM_GENERAL_FOCUS_BLOCK,
    OFFICIAL_CAMPUS_ADDRESS_BLOCK,
    _SAFE_FALLBACK_EN,
    _SAFE_FALLBACK_TR,
)
from chatbot.services.context_select import (
    _answer_is_stock_no_info,
    _context_likely_relevant,
    _select_context_for_llm,
    _strip_urls_plain_text,
)
from chatbot.services.conversation_repo import (
    build_assistant_reply,
    conversation_title_from_question,
)
from chatbot.services.embedding import _retrieve_top_chunks_by_embedding
from chatbot.services.extractive import _try_extractive_answer
from chatbot.services.intents import (
    _asks_subunits_of_named_faculty,
    _faculty_department_catalog_intent,
    _general_acibadem_intro_intent,
    _green_or_sustainable_campus_question,
    _is_extractive_question,
)
from chatbot.services.language import (
    _ascii_fold_turkish,
    _detect_language,
    _looks_english,
    _looks_turkish,
)
from chatbot.services.llm_client import (
    OLLAMA_TIMEOUT_SENTINEL,
    ask_gemma,
    translate_answer,
)
from chatbot.services.prompts import build_ask_prompt
from chatbot.services.query_parser import QueryFilters, parse_query

logger = logging.getLogger(__name__)

# answer_source values surfaced on the wire so the frontend can branch
# (e.g. show a "no context found" badge for FALLBACK without parsing the body).
ANSWER_SOURCE_RAG_LLM = "RAG_LLM"
ANSWER_SOURCE_EXTRACTIVE = "EXTRACTIVE"
ANSWER_SOURCE_FALLBACK = "FALLBACK"
ANSWER_SOURCE_NO_INFO = "NO_INFO"


@dataclass
class AskMeta:
    """Side-channel context produced by ``run_ask`` for view-layer serialization.

    Kept as a dataclass (not pre-serialized JSON) so the orchestrator stays
    HTTP-agnostic: views in :mod:`chatbot.api.v1` decide whether to expose
    ``retrieved_chunks`` text, what the wire shape is, etc. An empty instance
    is returned for early failures (no retrieval performed).
    """

    answer_source: str | None = None
    retrieved_chunks: list[dict] = field(default_factory=list)
    filters: QueryFilters = field(default_factory=QueryFilters)
    latency_ms: dict[str, int] = field(default_factory=dict)


def _context_from_hybrid_chunks(chunks: list[dict]) -> str:
    """Turn hybrid embedding hits into the same ``---`` block layout as legacy retrieval.

    Each block's first line is a ``[title | url]`` tag so :func:`_select_context_for_llm`
    and :func:`_extract_block_source_label` behave consistently.
    """
    blocks: list[str] = []
    for c in chunks:
        title = (c.get("title") or "").strip()
        url = (c.get("url") or "").strip()
        body = (c.get("text") or "").strip()
        if not body:
            continue
        meta_parts = [p for p in (title, url) if p]
        meta = " | ".join(meta_parts)
        blocks.append(f"[{meta}]\n{body}" if meta else body)
    return "\n\n---\n\n".join(blocks).strip()


def run_ask(question: str, conv) -> tuple[dict, int, AskMeta]:
    """Drive an /ask call from a sanitized question + resolved conversation.

    The HTTP adapter is expected to have:
      * parsed and validated the request body,
      * normalized the question (NFC),
      * dispatched any strict-rag-verify branch,
      * resolved/created the Conversation row.

    This function persists the user message, runs the retrieval/LLM pipeline, persists
    the assistant reply, and returns ``(payload, http_status, meta)``. ``meta`` is
    always a :class:`AskMeta` instance; views decide whether/how to put it on the
    wire (the legacy :func:`chatbot.api.v1.views.ask` ignores it; the v1
    endpoint serializes it into the response).
    """
    from chatbot.models import Message

    Message.objects.create(conversation=conv, role=Message.ROLE_USER, content=question)
    if not (conv.title or "").strip():
        conv.title = conversation_title_from_question(question)
        conv.save(update_fields=["title"])

    t_total_start = time.perf_counter()
    meta = AskMeta()

    def _finalize(
        reply: tuple[dict, int],
        *,
        answer_source: str | None,
    ) -> tuple[dict, int, AskMeta]:
        meta.answer_source = answer_source
        meta.latency_ms["total"] = int((time.perf_counter() - t_total_start) * 1000)
        return reply[0], reply[1], meta

    try:
        lang = _detect_language(question)
        is_tr = lang == "tr"
        no_info_msg = (
            "Bu konuda elimde net bir bilgi bulunamadı."
            if is_tr
            else "I couldn't find clear information about this."
        )

        # ASCII-fold so Turkish chars and .lower() quirks cannot skip intent detectors.
        q_fold = _ascii_fold_turkish(question)
        campus_green_q = _green_or_sustainable_campus_question(question)
        address_intent = (not campus_green_q) and any(
            t in q_fold
            for t in (
                "adres",
                "address",
                "kampus",
                "campus",
                "konum",
                "location",
                "ulasim",
                "nerede",
            )
        )
        dept_cat = _faculty_department_catalog_intent(question)
        sub_fac_units = _asks_subunits_of_named_faculty(question)
        general_intro = _general_acibadem_intro_intent(question)

        # Parser runs BEFORE k_ctx so the structured filters can drive retrieval
        # width. ``parse_query`` is pure (no DB / network), so moving it up here
        # is free.
        filters = parse_query(question)

        # Department-scoped course catalog: parser flags both a department AND
        # a BOLOGNA_COURSE/PROGRAM intent, but no specific course_code. These
        # questions ("Bilgisayar mühendisliği dersleri nedir?", "What courses
        # does Industrial Engineering have?") are inherently wide; the default
        # k_ctx=5 only surfaces program-overview chunks and the LLM refuses
        # with a stock "no info" reply. Adım 5.3 fix.
        wants_course_catalog = (
            bool(filters.department)
            and not filters.course_code
            and (
                ContentType.BOLOGNA_COURSE in filters.content_types
                or ContentType.BOLOGNA_PROGRAM in filters.content_types
            )
        )

        k_ctx = 5
        if address_intent or campus_green_q:
            k_ctx = 8
        if general_intro:
            k_ctx = max(k_ctx, 7)
        if sub_fac_units:
            k_ctx = max(k_ctx, 10)
        if wants_course_catalog:
            # 14 chunks is empirically enough for the LLM to enumerate ~10
            # courses while staying under DJANGO_MAX_CONTEXT_CHARS.
            k_ctx = max(k_ctx, 14)
        if dept_cat:
            # Fakülte tam listesi için daha fazla parça + bağlam sınırı (model yine kısaltabilir).
            k_ctx = max(k_ctx, 18)
        t_retrieve = time.perf_counter()
        chunks = _retrieve_top_chunks_by_embedding(question, k=k_ctx, filters=filters)
        retrieve_ms = int((time.perf_counter() - t_retrieve) * 1000)
        meta.filters = filters
        meta.retrieved_chunks = list(chunks)
        meta.latency_ms["retrieve"] = retrieve_ms
        context = _context_from_hybrid_chunks(chunks)
        logger.info(
            "/ask hybrid_retrieval done in %.2fs chunks=%s matched_terms=%s",
            retrieve_ms / 1000.0,
            len(chunks),
            filters.matched_terms,
        )
        if general_intro:
            ctx0 = (context or "").strip()
            context = (
                f"{ACIBADEM_GENERAL_FOCUS_BLOCK}\n\n{ctx0}".strip()
                if ctx0
                else ACIBADEM_GENERAL_FOCUS_BLOCK.strip()
            )
        if address_intent:
            ctx_body = (context or "").strip()
            context = (
                f"{OFFICIAL_CAMPUS_ADDRESS_BLOCK}\n{ctx_body}".strip()
                if ctx_body
                else OFFICIAL_CAMPUS_ADDRESS_BLOCK
            )
            # Small models latch onto random URLs from retrieved chunks; keep plain text for the model.
            context = re.sub(r"https?://\S+", "", context)
            context = re.sub(r" {2,}", " ", context)
            context = re.sub(r"\n{3,}", "\n\n", context).strip()
        selected_max_chunks = int(os.environ.get("DJANGO_SELECTED_MAX_CHUNKS", "4"))
        selected_max_chars = int(os.environ.get("DJANGO_SELECTED_MAX_CHARS", "4200"))
        if wants_course_catalog:
            # Course-list answers need many short chunks (one per course). The
            # default budget (4 chunks / 4200 chars) prunes us back to ~one
            # program-overview block, which is exactly what triggers the
            # "no info" stock reply on questions like
            # "Bilgisayar mühendisliği dersleri nedir?".
            selected_max_chunks = max(selected_max_chunks, 10)
            selected_max_chars = max(selected_max_chars, 7000)
        selected_context, selected_sources, retrieved_chunks = _select_context_for_llm(
            question,
            context,
            max_chunks=selected_max_chunks,
            max_chars=selected_max_chars,
        )
        context = selected_context
        logger.info("SELECTED_CONTEXT_FILES=%s", selected_sources)
        logger.info(
            "OLLAMA_PRECHECK question=%r retrieved_chunks=%s selected_sources=%s context_chars=%s",
            question,
            retrieved_chunks,
            selected_sources,
            len(context),
        )
        if not context:
            logger.info("ANSWER_SOURCE=FALLBACK")
            logger.info("EXTRACTIVE_REASON=context_weak_or_unrelated")
            return _finalize(
                build_assistant_reply(
                    conv,
                    _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
                ),
                answer_source=ANSWER_SOURCE_FALLBACK,
            )
        if _is_extractive_question(question):
            logger.info("EXTRACTIVE_ATTEMPTED")
            extractive = _try_extractive_answer(question, context)
            if extractive:
                answer_text, reason = extractive
                logger.info("ANSWER_SOURCE=EXTRACTIVE")
                logger.info("EXTRACTIVE_FOUND")
                logger.info("EXTRACTIVE_REASON=%s", reason)
                return _finalize(
                    build_assistant_reply(
                        conv,
                        answer_text,
                        attach_followup=False,
                        is_tr=is_tr,
                        question=question,
                    ),
                    answer_source=ANSWER_SOURCE_EXTRACTIVE,
                )
            logger.info("EXTRACTIVE_NOT_FOUND_CONTINUE_TO_LLM")

        max_context_chars = int(os.environ.get("DJANGO_MAX_CONTEXT_CHARS", "4200"))
        if dept_cat:
            max_context_chars = max(
                max_context_chars,
                int(os.environ.get("DJANGO_DEPT_CATALOG_CONTEXT_CHARS", "10000")),
            )
        if wants_course_catalog:
            # Mirror the bumped select-step budget so the post-cap doesn't
            # silently truncate everything we just paid retrieval cost for.
            max_context_chars = max(max_context_chars, selected_max_chars)
        if len(context) > max_context_chars:
            context = context[:max_context_chars].rsplit("\n", 1)[0].strip()

        prompt = build_ask_prompt(
            question=question,
            context=context,
            is_tr=is_tr,
            address_intent=address_intent,
            campus_green_q=campus_green_q,
            dept_cat=dept_cat,
            general_intro=general_intro,
        )
        logger.info(
            "OLLAMA_INPUT question=%r retrieved_chunks=%s selected_sources=%s context_chars=%s prompt_chars=%s",
            question,
            retrieved_chunks,
            selected_sources,
            len(context),
            len(prompt),
        )
        t_llm = time.perf_counter()
        answer = ask_gemma(prompt)
        llm_elapsed_ms = int((time.perf_counter() - t_llm) * 1000)
        meta.latency_ms["llm"] = meta.latency_ms.get("llm", 0) + llm_elapsed_ms
        logger.info("/ask ask_gemma (primary) done in %.2fs", llm_elapsed_ms / 1000.0)
        if answer == OLLAMA_TIMEOUT_SENTINEL:
            logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(prompt))
            logger.info("ANSWER_SOURCE=FALLBACK")
            return _finalize(
                build_assistant_reply(
                    conv,
                    _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
                ),
                answer_source=ANSWER_SOURCE_FALLBACK,
            )
        if (
            not (answer or "").strip()
            and len((context or "").strip()) > 180
            and retrieved_chunks >= 2
        ):
            refill = (
                "Aşağıdaki Bağlamı kullanarak soruyu Türkçe yanıtla. Boş bırakma; en az 2 anlamlı cümle yaz.\n\n"
                f"Bağlam:\n{context[:4000]}\n\nSoru:\n{question}"
                if is_tr
                else (
                    "Answer the question in English using the context below. Do not leave the answer empty; "
                    "at least 2 meaningful sentences.\n\n"
                    f"Context:\n{context[:4000]}\n\nQuestion:\n{question}"
                )
            )
            t_ref = time.perf_counter()
            answer = (ask_gemma(refill) or "").strip()
            ref_elapsed_ms = int((time.perf_counter() - t_ref) * 1000)
            meta.latency_ms["llm"] = meta.latency_ms.get("llm", 0) + ref_elapsed_ms
            logger.info("/ask ask_gemma empty-refill done in %.2fs", ref_elapsed_ms / 1000.0)
            if answer == OLLAMA_TIMEOUT_SENTINEL:
                logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(refill))
                logger.info("ANSWER_SOURCE=FALLBACK")
                return _finalize(
                    build_assistant_reply(
                        conv,
                        _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                        attach_followup=False,
                        is_tr=is_tr,
                        question=question,
                    ),
                    answer_source=ANSWER_SOURCE_FALLBACK,
                )
        ctx_lc = (context or "").lower()
        if campus_green_q and ctx_lc and any(
            w in ctx_lc
            for w in (
                "sürdürülebilir",
                "surdurulebilir",
                "sustainable",
                "iklim",
                "çevre",
                "cevre",
                "yeşil",
                "yesil",
                "karbon",
                "leed",
                "eko",
            )
        ):
            if _answer_is_stock_no_info(answer):
                retry = (
                    "Aşağıdaki bağlamdan YALNIZCA yazılanları kullanarak soruyu Türkçe, **3–6 kısa cümle** ile yanıtla. "
                    "Uydurma bilgi ekleme. Bağlamda sürdürülebilirlik, çevre veya kampüsle ilgili ne varsa açıkla.\n\n"
                    f"Bağlam:\n{context[:4000]}\n\nSoru: {question}"
                    if is_tr
                    else (
                        "Answer in English using ONLY the context below (**3–6 short sentences**). "
                        "Do not invent facts. Explain any sustainability, environment, or campus-related wording.\n\n"
                        f"Context:\n{context[:4000]}\n\nQuestion: {question}"
                    )
                )
                t_retry = time.perf_counter()
                retry_answer = (ask_gemma(retry) or "").strip()
                meta.latency_ms["llm"] = meta.latency_ms.get("llm", 0) + int(
                    (time.perf_counter() - t_retry) * 1000
                )
                if retry_answer == OLLAMA_TIMEOUT_SENTINEL:
                    logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(retry))
                    logger.info("ANSWER_SOURCE=FALLBACK")
                    return _finalize(
                        build_assistant_reply(
                            conv,
                            _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                            attach_followup=False,
                            is_tr=is_tr,
                            question=question,
                        ),
                        answer_source=ANSWER_SOURCE_FALLBACK,
                    )
                answer = retry_answer or answer
        generic_retry_on = (os.environ.get("DJANGO_ENABLE_GENERIC_RETRY") or "0").strip().lower() in (
            "1",
            "true",
            "yes",
        )
        if (
            generic_retry_on
            and (
                _context_likely_relevant(question, context)
                and _answer_is_stock_no_info(answer)
                and len((context or "").strip()) > 120
            )
        ):
            retry_gen = (
                "Kullanıcının sorusu ile aşağıdaki bağlam arasında anlamlı kelime örtüşmesi var. "
                "Bağlamdan YALNIZCA desteklenen bilgileri kullanarak Türkçe, **3–6 kısa cümle** yaz. "
                "Uydurma. Bağlam gerçekten cevap vermiyorsa tek cümlede 'Bu konuda elimde net bir bilgi bulunamadı.' de.\n\n"
                f"Bağlam:\n{context[:4200]}\n\nSoru: {question}"
                if is_tr
                else (
                    "There is lexical overlap between the question and the context below. "
                    "Write a helpful answer in English using ONLY supported facts (**3–6 short sentences**). "
                    "Do not invent. If the context truly does not support an answer, output only: "
                    "I couldn't find clear information about this.\n\n"
                    f"Context:\n{context[:4200]}\n\nQuestion: {question}"
                )
            )
            t_gen = time.perf_counter()
            retry_gen_answer = (ask_gemma(retry_gen) or "").strip()
            meta.latency_ms["llm"] = meta.latency_ms.get("llm", 0) + int(
                (time.perf_counter() - t_gen) * 1000
            )
            if retry_gen_answer == OLLAMA_TIMEOUT_SENTINEL:
                logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(retry_gen))
                logger.info("ANSWER_SOURCE=FALLBACK")
                return _finalize(
                    build_assistant_reply(
                        conv,
                        _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                        attach_followup=False,
                        is_tr=is_tr,
                        question=question,
                    ),
                    answer_source=ANSWER_SOURCE_FALLBACK,
                )
            answer = retry_gen_answer or answer
        if answer.startswith("Gemma error:"):
            # Error path: keep legacy ``detail`` shape; meta still travels for /api/v1/ask.
            return _finalize(
                build_assistant_reply(conv, answer, status=502, as_detail=True),
                answer_source=None,
            )
        if not (answer or "").strip():
            return _finalize(
                build_assistant_reply(
                    conv,
                    no_info_msg,
                    attach_followup=True,
                    is_tr=is_tr,
                    question=question,
                ),
                answer_source=ANSWER_SOURCE_NO_INFO,
            )
        # Dil düzeltmesi: ikinci bir tam Ollama çağrısı dakikalarca sürebilir. Türkçe soruda yalnızca
        # cevap belirgin İngilizceyse çevir (ASCII Türkçe yanıtı yanlışlıkla İngilizce sanma).
        if is_tr:
            if _looks_english(answer):
                t_tr = time.perf_counter()
                answer = translate_answer(answer, "tr")
                logger.info("/ask translate->tr done in %.2fs", time.perf_counter() - t_tr)
        else:
            if _looks_turkish(answer) and not _looks_english(answer):
                t_tr = time.perf_counter()
                answer = translate_answer(answer, "en")
                logger.info("/ask translate->en done in %.2fs", time.perf_counter() - t_tr)
        if address_intent:
            answer = _strip_urls_plain_text(answer)
        logger.info("ANSWER_SOURCE=RAG_LLM")
        return _finalize(
            build_assistant_reply(
                conv,
                answer,
                attach_followup=True,
                is_tr=is_tr,
                question=question,
            ),
            answer_source=ANSWER_SOURCE_RAG_LLM,
        )
    except Exception:
        logger.exception("Failed to answer question in /ask")
        err_text = "Backend initialization failed. Check model/dependencies and server logs."
        return _finalize(
            build_assistant_reply(conv, err_text, status=500, as_detail=True),
            answer_source=None,
        )


__all__ = ["run_ask", "AskMeta", "ANSWER_SOURCE_RAG_LLM", "ANSWER_SOURCE_EXTRACTIVE",
           "ANSWER_SOURCE_FALLBACK", "ANSWER_SOURCE_NO_INFO"]

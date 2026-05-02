"""End-to-end /ask pipeline.

This is the only services-layer module allowed to know that an /ask call has many
stages (intents → retrieval → LLM → post-processing → persistence). It stays
HTTP-agnostic: returns ``(payload_dict, status_code)`` for the HTTP adapter to
wrap into a JsonResponse.

Allowed dependency direction (top of the DAG):
    ask_orchestrator → constants, language, intents, context_select, extractive,
                       retrieval, embedding, llm_client, prompts, conversation_repo
"""
from __future__ import annotations

import logging
import os
import re
import time

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
    _thread_suggests_acibadem_topic,
)
from chatbot.services.conversation_repo import (
    build_assistant_reply,
    conversation_title_from_question,
)
from chatbot.services.embedding import _retrieve_top_chunks_by_embedding
from chatbot.services.extractive import _try_extractive_answer
from chatbot.services.intents import (
    _asks_subunits_of_named_faculty,
    _canonical_campus_address_reply,
    _ce_overview_context_block,
    _cs_engineering_course_catalog_intent,
    _cs_engineering_lisans_intent,
    _engineering_faculty_departments_intent,
    _engineering_faculty_departments_reply,
    _faculty_department_catalog_intent,
    _general_acibadem_intro_intent,
    _green_or_sustainable_campus_question,
    _is_extractive_question,
    _wants_postal_address_detail,
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
from chatbot.services.retrieval import retrieve_context

logger = logging.getLogger(__name__)


def run_ask(question: str, conv) -> tuple[dict, int]:
    """Drive an /ask call from a sanitized question + resolved conversation.

    The HTTP adapter is expected to have:
      * parsed and validated the request body,
      * normalized the question (NFC),
      * dispatched any strict-rag-verify branch,
      * resolved/created the Conversation row.

    This function persists the user message, runs the retrieval/LLM pipeline, persists
    the assistant reply, and returns ``(payload, http_status)``.
    """
    from chatbot.models import Message

    Message.objects.create(conversation=conv, role=Message.ROLE_USER, content=question)
    if not (conv.title or "").strip():
        conv.title = conversation_title_from_question(question)
        conv.save(update_fields=["title"])

    try:
        lang = _detect_language(question)
        is_tr = lang == "tr"
        no_info_msg = (
            "Bu konuda elimde net bir bilgi bulunamadı."
            if is_tr
            else "I couldn't find clear information about this."
        )

        if _engineering_faculty_departments_intent(question):
            return build_assistant_reply(
                conv,
                _engineering_faculty_departments_reply(),
                attach_followup=False,
                is_tr=True,
                question=question,
            )

        # ASCII-fold so Turkish chars and .lower() quirks cannot skip the postal shortcut.
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
        if _wants_postal_address_detail(q_fold) and (
            "acibadem" in q_fold or _thread_suggests_acibadem_topic(conv)
        ):
            return build_assistant_reply(
                conv,
                _canonical_campus_address_reply(is_tr),
                attach_followup=True,
                is_tr=is_tr,
                question=question,
            )

        cs_eng_q = _cs_engineering_lisans_intent(question)
        cs_course_catalog_q = _cs_engineering_course_catalog_intent(question)
        dept_cat = _faculty_department_catalog_intent(question)
        sub_fac_units = _asks_subunits_of_named_faculty(question)
        general_intro = _general_acibadem_intro_intent(question)
        k_ctx = 5
        if address_intent or cs_eng_q or campus_green_q:
            k_ctx = 8
        if general_intro:
            k_ctx = max(k_ctx, 7)
        if sub_fac_units:
            k_ctx = max(k_ctx, 10)
        if dept_cat:
            # Fakülte tam listesi için daha fazla parça + bağlam sınırı (model yine kısaltabilir).
            k_ctx = max(k_ctx, 18)
        if cs_course_catalog_q:
            k_ctx = max(k_ctx, 14)
        t_retrieve = time.perf_counter()
        context = retrieve_context(question, k=k_ctx)
        logger.info("/ask retrieve_context done in %.2fs", time.perf_counter() - t_retrieve)
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
        if cs_eng_q and not cs_course_catalog_q:
            ce_block = _ce_overview_context_block()
            ctx_body = (context or "").strip()
            context = f"{ce_block}\n\n{ctx_body}".strip() if ctx_body else ce_block
        # Varsayılan KAPALI: CPU'da tüm embedding matrisi + Ollama 4–7 dk'ı aşabiliyor.
        # Açmak için: ACU_COURSE_CATALOG_EMBED_AUGMENT=1 (veya true) — önbellek ısındıktan sonra daha hızlı.
        if cs_course_catalog_q and (
            (os.environ.get("ACU_COURSE_CATALOG_EMBED_AUGMENT") or "0").strip().lower()
            not in ("0", "false", "no")
        ):
            try:
                from chatbot.models import ChunkEmbedding as _CEMod

                emb_k = min(8, max(6, k_ctx // 2 + 2))
                obs_emb_n = _CEMod.objects.filter(chunk__source_type="obs").count()
                # Tek tarama: OBS embedding varsa sadece obs; yoksa tümü (çift tarama kaldırıldı).
                emb_chunks = _retrieve_top_chunks_by_embedding(
                    question,
                    k=emb_k,
                    source_type="obs" if obs_emb_n > 0 else None,
                )
                if emb_chunks:
                    lines: list[str] = []
                    for c in emb_chunks:
                        meta = " | ".join([p for p in [c.get("title") or "", c.get("url") or ""] if p])
                        t = (c.get("text") or "").strip()
                        if not t:
                            continue
                        lines.append(f"[{meta}]\n{t}" if meta else t)
                    if lines:
                        inject = "\n\n---\n\n".join(lines)
                        base = (context or "").strip()
                        context = (
                            f"{base}\n\n---\n\n"
                            f"[İlgili parçalar — anlamsal (embedding) arama]\n\n{inject}".strip()
                            if base
                            else f"[İlgili parçalar — anlamsal (embedding) arama]\n\n{inject}".strip()
                        )
            except Exception:
                logger.exception("course_catalog_embedding_augment_failed")
        selected_context, selected_sources, retrieved_chunks = _select_context_for_llm(
            question,
            context,
            max_chunks=int(os.environ.get("DJANGO_SELECTED_MAX_CHUNKS", "4")),
            max_chars=int(os.environ.get("DJANGO_SELECTED_MAX_CHARS", "4200")),
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
            return build_assistant_reply(
                conv,
                _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                attach_followup=False,
                is_tr=is_tr,
                question=question,
            )
        if _is_extractive_question(question):
            logger.info("EXTRACTIVE_ATTEMPTED")
            extractive = _try_extractive_answer(question, context)
            if extractive:
                answer_text, reason = extractive
                logger.info("ANSWER_SOURCE=EXTRACTIVE")
                logger.info("EXTRACTIVE_FOUND")
                logger.info("EXTRACTIVE_REASON=%s", reason)
                return build_assistant_reply(
                    conv,
                    answer_text,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
                )
            logger.info("EXTRACTIVE_NOT_FOUND_CONTINUE_TO_LLM")

        max_context_chars = int(os.environ.get("DJANGO_MAX_CONTEXT_CHARS", "4200"))
        embed_augment_on = (
            (os.environ.get("ACU_COURSE_CATALOG_EMBED_AUGMENT") or "0").strip().lower()
            not in ("0", "false", "no")
        )
        if dept_cat:
            max_context_chars = max(
                max_context_chars,
                int(os.environ.get("DJANGO_DEPT_CATALOG_CONTEXT_CHARS", "10000")),
            )
        if cs_course_catalog_q and embed_augment_on:
            max_context_chars = max(
                max_context_chars,
                int(os.environ.get("DJANGO_COURSE_CATALOG_CONTEXT_CHARS", "14000")),
            )
        if len(context) > max_context_chars:
            context = context[:max_context_chars].rsplit("\n", 1)[0].strip()

        prompt = build_ask_prompt(
            question=question,
            context=context,
            is_tr=is_tr,
            address_intent=address_intent,
            cs_eng_q=cs_eng_q,
            cs_course_catalog_q=cs_course_catalog_q,
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
        logger.info("/ask ask_gemma (primary) done in %.2fs", time.perf_counter() - t_llm)
        if answer == OLLAMA_TIMEOUT_SENTINEL:
            logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(prompt))
            logger.info("ANSWER_SOURCE=FALLBACK")
            return build_assistant_reply(
                conv,
                _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                attach_followup=False,
                is_tr=is_tr,
                question=question,
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
            logger.info("/ask ask_gemma empty-refill done in %.2fs", time.perf_counter() - t_ref)
            if answer == OLLAMA_TIMEOUT_SENTINEL:
                logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(refill))
                logger.info("ANSWER_SOURCE=FALLBACK")
                return build_assistant_reply(
                    conv,
                    _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
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
                retry_answer = (ask_gemma(retry) or "").strip()
                if retry_answer == OLLAMA_TIMEOUT_SENTINEL:
                    logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(retry))
                    logger.info("ANSWER_SOURCE=FALLBACK")
                    return build_assistant_reply(
                        conv,
                        _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                        attach_followup=False,
                        is_tr=is_tr,
                        question=question,
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
            retry_gen_answer = (ask_gemma(retry_gen) or "").strip()
            if retry_gen_answer == OLLAMA_TIMEOUT_SENTINEL:
                logger.info("OLLAMA_TIMEOUT prompt_chars=%s", len(retry_gen))
                logger.info("ANSWER_SOURCE=FALLBACK")
                return build_assistant_reply(
                    conv,
                    _SAFE_FALLBACK_TR if is_tr else _SAFE_FALLBACK_EN,
                    attach_followup=False,
                    is_tr=is_tr,
                    question=question,
                )
            answer = retry_gen_answer or answer
        if answer.startswith("Gemma error:"):
            return build_assistant_reply(conv, answer, status=502, as_detail=True)
        if not (answer or "").strip():
            return build_assistant_reply(
                conv,
                no_info_msg,
                attach_followup=True,
                is_tr=is_tr,
                question=question,
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
        return build_assistant_reply(
            conv,
            answer,
            attach_followup=True,
            is_tr=is_tr,
            question=question,
        )
    except Exception:
        logger.exception("Failed to answer question in /ask")
        err_text = "Backend initialization failed. Check model/dependencies and server logs."
        return build_assistant_reply(conv, err_text, status=500, as_detail=True)


__all__ = ["run_ask"]

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

import requests
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)


_TR_STOPWORDS = {
    "ve",
    "veya",
    "ile",
    "için",
    "icin",
    "nedir",
    "ne",
    "nerede",
    "nasıl",
    "nasil",
    "hangi",
    "kim",
    "kaç",
    "kac",
    "mı",
    "mi",
    "mu",
    "mü",
    "bu",
    "şu",
    "su",
    "o",
    "bir",
    "da",
    "de",
    "ama",
    "fakat",
    "gibi",
    "üniversite",
    "universite",
    "üniversitesi",
    "universitesi",
    "acibadem",
    "acıbadem",
}


def _looks_acibadem_related(question: str) -> bool:
    q = (question or "").lower()
    return any(
        k in q
        for k in (
            "acıbadem",
            "acibadem",
            "acıbadem üniversitesi",
            "acibadem universitesi",
            "acibadem university",
        )
    )


def _detect_language(question: str) -> str:
    """
    Heuristic language detection for TR vs EN.
    Returns: "tr" or "en"
    """
    q = (question or "").strip().lower()
    if not q:
        return "tr"

    # Turkish-specific characters are a strong signal.
    if any(ch in q for ch in ("ç", "ğ", "ı", "ö", "ş", "ü")):
        return "tr"

    tr_markers = {
        "mı",
        "mi",
        "mu",
        "mü",
        "nedir",
        "nerede",
        "nasıl",
        "nasil",
        "kimdir",
        "hangisi",
        "anlat",
        "fakülte",
        "fakulte",
        "bölüm",
        "bolum",
        "üniversite",
        "universite",
        "adres",
        "ulaşım",
        "ulasim",
    }
    en_markers = {
        "what",
        "where",
        "how",
        "who",
        "tell",
        "explain",
        "faculty",
        "department",
        "university",
        "address",
        "campus",
        "admission",
        "apply",
    }

    tokens = set(q.replace("?", " ").replace(".", " ").replace(",", " ").split())
    if tokens & tr_markers:
        return "tr"
    if tokens & en_markers:
        return "en"

    # Default: Turkish (project is TR-first).
    return "tr"


def _looks_turkish(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    if any(ch in t for ch in ("ç", "ğ", "ı", "ö", "ş", "ü")):
        return True
    # Common Turkish function words
    tr_words = (" ve ", " ile ", " için ", " adres", " üniversite", " kampüs", " istanbul", " türkiye", " nedir", " nerede")
    hits = sum(1 for w in tr_words if w in f" {t} ")
    return hits >= 2


def _looks_english(text: str) -> bool:
    t = (text or "").lower()
    if not t:
        return False
    en_words = (" the ", " and ", " address", " university", " campus", " contact", " located", " is ")
    hits = sum(1 for w in en_words if w in f" {t} ")
    return hits >= 2


def _translate_answer(answer: str, target_lang: str) -> str:
    """
    Translate an already-produced answer to the target language ("tr" or "en").
    Must preserve meaning and avoid adding new facts.
    """
    if target_lang not in ("tr", "en"):
        return answer
    if not (answer or "").strip():
        return answer
    to_label = "Turkish" if target_lang == "tr" else "English"
    prompt = f"""Translate the text below to {to_label}.
Rules:
- Preserve meaning exactly.
- Do not add any new information.
- Output only the translation (no quotes, no extra text).

Text:
{answer}
"""
    translated = ask_gemma(prompt)
    return (translated or "").strip() or answer

def _extract_keywords(question: str) -> list[str]:
    tokens = [
        t.strip(".,!?;:()[]{}\"'").lower()
        for t in (question or "").split()
        if len(t.strip(".,!?;:()[]{}\"'")) >= 3
    ]
    keywords = [t for t in tokens if t not in _TR_STOPWORDS]
    # keep order, de-dup
    seen: set[str] = set()
    out: list[str] = []
    for k in keywords:
        if k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out


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
    `RETRIEVE_MAX_CANDIDATES` (default 6000). If no keyword matches, the most recently updated
    chunks up to that cap are used instead.
    """
    keywords = _extract_keywords(question)
    q_lower = (question or "").lower()

    # Intent detection (priority matters):
    # - If question mentions staj/internship => internship intent
    # - Else if question mentions apply/admission terms => general admission intent
    internship_intent = any(t in q_lower for t in ("staj", "internship"))
    admission_intent = (not internship_intent) and any(
        t in q_lower for t in ("başvuru", "basvuru", "admission", "apply", "application")
    )
    address_intent = any(
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

    def _score_row(row) -> int:
        title = (row.title or "").lower()
        section = (row.section or "").lower()
        text = (row.chunk_text or "").lower()
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

        return score

    # 1) Prefer DB chunks if available — score a bounded candidate set, then take top k by score.
    try:
        from chatbot.models import PageChunk

        if PageChunk.objects.exists():
            max_candidates = int(os.environ.get("RETRIEVE_MAX_CANDIDATES", "6000"))
            base_qs = PageChunk.objects.only(
                "chunk_text", "title", "section", "url", "source_type", "updated_at"
            ).order_by("-updated_at")

            lookup_terms: list[str] = []
            seen_terms: set[str] = set()
            for kw in keywords[:12]:
                if len(kw) < 3:
                    continue
                for term in (kw, kw[:5] if len(kw) >= 7 else ""):
                    if len(term) >= 3 and term not in seen_terms:
                        seen_terms.add(term)
                        lookup_terms.append(term)
                if len(lookup_terms) >= 18:
                    break

            candidate_qs = base_qs
            if lookup_terms:
                q_filter = Q()
                for term in lookup_terms:
                    q_filter |= (
                        Q(chunk_text__icontains=term)
                        | Q(title__icontains=term)
                        | Q(section__icontains=term)
                        | Q(url__icontains=term)
                    )
                narrowed = base_qs.filter(q_filter)
                candidate_qs = narrowed if narrowed.exists() else base_qs[:max_candidates]
            else:
                candidate_qs = base_qs[:max_candidates]

            candidate_qs = candidate_qs[:max_candidates]

            scored_all: list[tuple[int, object, object]] = []
            for row in candidate_qs.iterator(chunk_size=500):
                s = _score_row(row)
                updated = row.updated_at or ""
                scored_all.append((s, updated, row))

            scored_all.sort(key=lambda x: (x[0], x[1]), reverse=True)
            top_score = scored_all[0][0] if scored_all else 0
            if top_score <= 0:
                return ""

            logger.info("retrieve_context(db): question=%r candidates=%s", question, len(scored_all))
            for rank, (s, _u, row) in enumerate(scored_all[:k], start=1):
                logger.info("retrieve_context(db): #%s score=%s title=%r", rank, s, row.title)

            top_rows = [row for _, _, row in scored_all[:k]]
            blocks: list[str] = []
            for c in top_rows:
                meta = " | ".join(
                    [p for p in [c.title or "", c.section or "", c.source_type or "", c.url or ""] if p]
                )
                blocks.append(f"[{meta}]\n{c.chunk_text}" if meta else c.chunk_text)

            return "\n\n---\n\n".join(blocks).strip()
    except Exception:
        pass

    # 2) Fallback: local /data/*.txt — include all scored lines, take top k even if low score.
    try:
        from rag.document_loader import load_text_documents
        from rag.text_splitter import split_into_chunks
    except Exception:
        return ""

    base_dir = Path(__file__).resolve().parent.parent
    data_dir = str(base_dir / "data")
    try:
        docs = load_text_documents(data_dir)
        chunks = split_into_chunks(docs, chunk_size=900, chunk_overlap=150)
    except Exception:
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



def ask_gemma(prompt: str) -> str:
    try:
        base_url = (os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434").rstrip("/")
        model = (os.environ.get("OLLAMA_MODEL") or "gemma2:2b").strip()
        # Keep responses fast/compact by default; can be overridden via env.
        num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "256"))
        temperature = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
        response = requests.post(
            f"{base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": num_predict,
                    "temperature": temperature,
                },
            },
            timeout=300,
        )
        if response.status_code == 404:
            body = (response.text or "").strip()[:400]
            return (
                "Gemma error: Ollama returned 404 (usually the model is not downloaded yet). "
                f"Run once: docker compose exec ollama ollama pull {model} "
                f"(or locally: ollama pull {model}). "
                f"Ollama said: {body or 'empty body'}"
            )
        response.raise_for_status()
        data = response.json()
        return (data["response"] or "").strip()
    except KeyError:
        return "Gemma error: Missing 'response' field in Ollama JSON."
    except Exception as e:
        return f"Gemma error: {str(e)}"


@lru_cache(maxsize=1)
def _bootstrap_rag() -> None:
    """RAG sistemini (yapay zekayı) bir kez yükler."""
    from backend.api import init_rag
    init_rag(model_path=os.environ.get("MODEL_PATH") or None)

def health(_request):
    return JsonResponse({"status": "ok"})

@csrf_exempt
@require_http_methods(["GET", "POST"])
def ask(request):
    # --- 1. ADIM: Arkadaşının Tasarımını Göster (Tarayıcıdan girince) ---
    if request.method == "GET":
        return render(request, "index.html") 

    # --- 2. ADIM: Gelen Mesajı Gemma'ya Gönder (Chat kutusuna yazınca) ---
    try:
        body = json.loads(request.body.decode("utf-8") or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"detail": "Invalid JSON."}, status=400)

    question = (body.get("question") or "").strip()
    if not question:
        return JsonResponse({"detail": "Question cannot be empty."}, status=400)

    try:
        lang = _detect_language(question)
        is_tr = lang == "tr"
        no_info_msg = (
            "Bu konuda elimde net bir bilgi bulunamadı."
            if is_tr
            else "I couldn't find clear information about this."
        )

        q_lower = question.lower()
        address_intent = any(
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
            )
        )

        context = retrieve_context(question, k=5)
        # If we cannot retrieve any relevant context and the query doesn't look on-topic,
        # avoid hallucinations by refusing politely.
        if not context and not _looks_acibadem_related(question):
            return JsonResponse(
                {
                    "answer": (
                        "Ben Acıbadem Üniversitesi odaklı bir asistanım."
                        if is_tr
                        else "I'm an assistant focused on Acibadem University."
                    )
                }
            )

        if not context:
            return JsonResponse(
                {
                    "answer": (
                        (
                            "Bu konuda güvenilir bilgi bulamadım. "
                            "Veritabanımda ilgili içerik yoksa uydurma bilgi veremem. "
                            "Resmî web sitesindeki İletişim/Ulaşım bölümünden doğrulayabilirsin."
                        )
                        if is_tr
                        else (
                            "I couldn't find reliable information in my data. "
                            "If it's not in my database, I can't make it up. "
                            "Please verify on the official website's Contact/Transportation pages."
                        )
                    )
                }
            )

        # If user asks for address/location but the retrieved context doesn't contain address-like signals,
        # don't waste time calling the model on irrelevant content.
        if address_intent:
            ctx_l = context.lower()
            has_address_signal = any(
                s in ctx_l
                for s in (
                    "istanbul",
                    "ataşehir",
                    "atasehir",
                    "cad",
                    "caddesi",
                    "sok",
                    "sk",
                    "no:",
                    "kampüs",
                    "kampus",
                    "yerleşke",
                    "yerleske",
                    "ulaşım",
                    "ulasim",
                    "iletişim",
                    "iletisim",
                )
            )
            if not has_address_signal:
                return JsonResponse(
                    {
                        "answer": (
                            (
                                "Adres/konum bilgisini veritabanımda net olarak bulamadım. "
                                "Yanlış bilgi vermemek için tahmin edemiyorum. "
                                "Resmî web sitesindeki İletişim/Ulaşım sayfasından kontrol edebilirsin."
                            )
                            if is_tr
                            else (
                                "I couldn't find a clear address/location in my database. "
                                "To avoid giving wrong information, I can't guess. "
                                "Please check the official website's Contact/Transportation page."
                            )
                        )
                    }
                )

        # Truncate context to keep latency manageable on small local models.
        max_context_chars = int(os.environ.get("DJANGO_MAX_CONTEXT_CHARS", "4500"))
        if len(context) > max_context_chars:
            context = context[:max_context_chars].rsplit("\n", 1)[0].strip()

        answer_language_instruction = "Türkçe" if is_tr else "English"

        prompt = f"""
You are a helpful university assistant.

LANGUAGE RULES:
- Detect the language of the user's question.
- Always answer in the SAME language as the user’s question.
- If the question is in English → answer in English.
- If the question is in Turkish → answer in Turkish.

RETRIEVAL RULES:
- The context may be in Turkish or English.
- You MUST use the context even if it is in a different language than the question.
- If needed, translate the relevant information before answering.

ANSWERING RULES:
- Do NOT hallucinate.
- If the answer exists in the context, use it clearly.
- If the context is in another language, translate it to the user’s language.
- If no relevant information exists, say exactly:
  - Turkish: "Bu konuda elimde net bir bilgi bulunamadı."
  - English: "I couldn't find clear information about this."

PRIORITY:
1. Use context
2. Translate if necessary
3. Answer in user's language

Important: The user's question language is {answer_language_instruction}. Your final answer must be in {answer_language_instruction}.

Bağlam:
{context}

Kullanıcı sorusu:
{question}
 
 Cevap (yalnızca bağlama dayanarak):
"""
        answer = ask_gemma(prompt)
        if answer.startswith("Gemma error:"):
            return JsonResponse({"detail": answer}, status=502)
        if not (answer or "").strip():
            return JsonResponse({"answer": no_info_msg})
        # Enforce answer language: if model drifts (or mixes languages), translate back
        # without adding facts. Be strict for EN questions (UI expectation).
        if is_tr:
            if (not _looks_turkish(answer)) or _looks_english(answer):
                answer = _translate_answer(answer, "tr")
        else:
            if (not _looks_english(answer)) or _looks_turkish(answer):
                answer = _translate_answer(answer, "en")
        return JsonResponse({"answer": answer})
    except Exception:
        logger.exception("Failed to answer question in /ask")
        return JsonResponse(
            {"detail": "Backend initialization failed. Check model/dependencies and server logs."},
            status=500,
        )
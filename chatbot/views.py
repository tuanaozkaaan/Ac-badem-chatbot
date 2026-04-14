import json
import logging
import os
from functools import lru_cache

from django.http import JsonResponse
from django.shortcuts import render  # Sayfayı göstermek için şart
from django.views.decorators.csrf import csrf_exempt  # Hatanın çözümü burada
from django.views.decorators.http import require_http_methods

logger = logging.getLogger(__name__)

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
        _bootstrap_rag()
        from backend.api import get_rag

        rag = get_rag()
        answer = rag.answer(question)
        return JsonResponse({"answer": answer})
    except Exception:
        logger.exception("Failed to answer question in /ask")
        return JsonResponse(
            {"detail": "Backend initialization failed. Check model/dependencies and server logs."},
            status=500,
        )
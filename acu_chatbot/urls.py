from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    # Legacy SPA + JSON routes (cookie-based CSRF, GET / serves the chat UI).
    path("", include("chatbot.urls")),
    # Backwards-compatible /api/* alias for older clients that assumed an API prefix.
    # Same view module as legacy; SPA template still served on GET /api/.
    path("api/", include("chatbot.urls")),
    # Adım 5.1: API-only surface. JSON in/out, no SPA, CSRF-exempt /ask for the
    # Next.js Route Handler proxy. Wire shape is documented in docs/openapi.yaml.
    path("api/v1/", include("chatbot.api.v1.urls")),
]

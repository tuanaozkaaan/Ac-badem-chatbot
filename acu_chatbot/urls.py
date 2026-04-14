from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("chatbot.urls")),
    # Also expose chatbot endpoints under /api/* for clients that assume an API prefix.
    path("api/", include("chatbot.urls")),
]

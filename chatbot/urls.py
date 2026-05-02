from django.urls import path

from chatbot.api.v1 import views

# URL string contract: kept identical to the legacy layout so the frontend (and any
# external clients) keep working unchanged. The view callables now live in
# ``chatbot.api.v1.views``; ``chatbot.views`` remains as a re-export shim.
urlpatterns = [
    path("conversations/", views.conversations_root),
    path("conversations/<int:pk>/", views.conversations_detail),
    # Serve the UI on the site root as well.
    path("", views.ask),
    # Support both /health and /health/ (Django APPEND_SLASH is False).
    path("health", views.health),
    path("health/", views.health),
    path("ask", views.ask),
    path("ask/", views.ask),
]

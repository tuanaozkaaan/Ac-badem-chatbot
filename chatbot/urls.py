from django.urls import path

from . import views

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

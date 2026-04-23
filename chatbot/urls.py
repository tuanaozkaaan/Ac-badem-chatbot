from django.urls import path

from . import views

urlpatterns = [
    # Serve the UI on the site root as well.
    path("", views.ask),
    # Support both /health and /health/ (Django APPEND_SLASH is False).
    path("health", views.health),
    path("health/", views.health),
    path("ask", views.ask),
    path("ask/", views.ask),
]

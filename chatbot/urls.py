from django.urls import path

from . import views

urlpatterns = [
    # Support both /health and /health/ (Django APPEND_SLASH is False).
    path("health", views.health),
    path("health/", views.health),
    path("ask", views.ask),
]

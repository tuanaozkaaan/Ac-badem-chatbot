"""Domain services layer for the chatbot app.

Modules in this package must remain HTTP-agnostic: no Django ``HttpRequest``,
``JsonResponse``, ``csrf_exempt``, or view decorators. The HTTP boundary
(``chatbot.api``) is the only allowed consumer of these services.
"""

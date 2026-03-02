from django.urls import path

from qbrain.views.demo import RunDemo

app_name = 'world'
urlpatterns = [
    path('demo/', RunDemo.as_view()),
]
# Optional: stripe webhook (requires stripe + fb_core); skip when not installed (e.g. testing)
try:
    from qbrain.views.stripe_webhook import stripe_webhook
    urlpatterns.append(path('webhook/', stripe_webhook))
except Exception:
    pass

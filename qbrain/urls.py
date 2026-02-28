from django.urls import path

from qbrain.views.demo import RunDemo
from qbrain.views.stripe_webhook import stripe_webhook

app_name = 'world'
urlpatterns = [
    path('demo/', RunDemo.as_view()),
    path('webhook/', stripe_webhook),
]

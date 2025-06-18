from django.urls import path

from qf_sim.dj.views.world import CreateWorldView

app_name = 'db'
urlpatterns = [
    path('upsert/', CreateWorldView.as_view()),
]

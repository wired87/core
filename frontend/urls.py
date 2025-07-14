from django.urls import path

from frontend.main import ChatView
from qf_sim.dj.views.demo import RunDemo
from qf_sim.dj.views.create_world import CreateWorldView

app_name = 'unicorn'
urlpatterns = [
    path('main/', ChatView.as_view()),
]

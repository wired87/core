from django.urls import path

from graph.dj.visual import GraphLookup
from graph.dj.brain_test import BrainTestView

app_name = "graph"
urlpatterns = [
    # client
    path('view/', GraphLookup.as_view()),
    path('brain/test/', BrainTestView.as_view()),
]


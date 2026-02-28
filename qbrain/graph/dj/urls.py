from django.urls import path

from qbrain.graph.dj.visual import GraphLookup
from qbrain.graph.dj.brain_test import BrainTestView

app_name = "graph"
urlpatterns = [
    # client
    path('view/', GraphLookup.as_view()),
    path('brain/test/', BrainTestView.as_view()),
]


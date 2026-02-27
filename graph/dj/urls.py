from django.urls import path

from graph.dj.visual import GraphLookup

app_name = "graph"
urlpatterns = [
    # client
    path('view/', GraphLookup.as_view()),
]


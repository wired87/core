from django.urls import path

from gke.views.delete_pod import DeletePodView

app_name = 'gke'
urlpatterns = [
    path('delete-pod/', DeletePodView.as_view()),
]

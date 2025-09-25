from django.urls import path

from cloud_batch.views import JobDetailsView, JobDeleteView, JobCreatorView
app_name = "batch"
urlpatterns = [
    path('create/', JobCreatorView.as_view(), name='job-create'),
    path('delete/', JobDeleteView.as_view(), name='job-create'),
    path('details/', JobDetailsView.as_view(), name='job-details'),
]
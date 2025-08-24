#
from django.urls import path

from documentai.invoice_view import InvoiceExtractorView
from documentai.views.upload import DocumentUploadView

app_name = "documentai"
urlpatterns = [
    path('up/', DocumentUploadView.as_view()),
    path('inv/', InvoiceExtractorView.as_view()),
]


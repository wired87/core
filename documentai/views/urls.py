#
from django.urls import path

from _google.documentai.invoice_view import InvoiceExtractorView
from _google.documentai.views.upload import DocumentUploadView

app_name = "docai"
urlpatterns = [
    path('up/', DocumentUploadView.as_view()),
    path('inv/', InvoiceExtractorView.as_view()),
]


"""
Now also use th eprovided classes to write a auery view to aske documents with generative ai
same style of the view. The drf class based view need to get a list of files as entry.  check properly if they do exist in documentai. if not, you will upload them using the pre defiend class above. All fiels the view received need to be extracted the content from and given to 
"""
import pandas as pd
from django.http import HttpResponse
from rest_framework import serializers
from rest_framework.response import Response
from rest_framework.views import APIView

from documentai.extractor import get_content, extract_keys, convert_rows


class ExtractedInvoiceSerializer(serializers.Serializer):
    attachments = serializers.ListField(
    child=serializers.FileField(),
)




class InvoiceExtractorView(APIView):
    serializer_class = ExtractedInvoiceSerializer

    def post(self, request, *args, **kwargs):
        attachments = request.FILES.getlist("attachments")

        if not attachments:
            return Response({"error": "No attachments provided."}, status=400)
        whole_content = get_content(attachments)
        keys = extract_keys(whole_content)
        rows: list[list] = convert_rows(whole_content, keys)
        df = pd.DataFrame(data=rows, columns=keys)

        # Spaltenreihenfolge stabil halten
        csv_data = df.to_csv(index=False)

        response = HttpResponse(csv_data, content_type="text/csv")
        response['Content-Disposition'] = 'attachment; filename="invoice_data.csv"'
        return response

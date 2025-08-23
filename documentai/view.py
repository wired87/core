from datetime import date

from _g_storage import MOUNT_PATH
from rest_framework.response import Response
from rest_framework.views import APIView

from _google import GCP_ID
from _google.documentai.proces_doc import process_document_sample



#max 15 Seiten

class DocumentView(APIView):

    def post(self, request, *args, **kwargs):
        self.logger.info("Request received")
        if not attachments:
            self.logger.info("Invalid Payload")
            return Response({"error":"No attachments provided."})
        self.logger.info("Attachments recieved")
        whole_content = get_content(attachments)
        keys = extract_keys(whole_content)
        rows: list[list] = convert_rows(whole_content, keys)
        self.logger.info("Attachments extracted")
        df = pd.DataFrame(data=rows, columns=keys)
        # Spaltenreihenfolge stabil halten
        csv_data = df.to_csv(index=False)

        self.logger.info("CSV created")

        return Response(
            content=csv_data,
            media_type="text/csv",
            headers={"Content-Disposition": 'attachment; filename="invoice_data.csv"'}
        )
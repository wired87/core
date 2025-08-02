from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from _g_storage import MAIN_BUCKET
from _g_storage.storage import GBucket


class GBucketAPIView(APIView):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bucket_manager = GBucket(bucket_name=MAIN_BUCKET)

    def post(self, request, format=None):
        """
        Lädt eine Datei in den Bucket hoch.
        """
        file_obj = request.FILES.get('file')
        dest_path = request.data.get('destination_path')

        if not file_obj or not dest_path:
            return Response({"error": "Datei und Zielpfad sind erforderlich."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            self.bucket_manager.upload_file(file_obj.temporary_file_path(), dest_path)
            return Response({"message": f"Datei {file_obj.name} erfolgreich hochgeladen."}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request, format=None):
        """
        Lädt eine Datei vom Bucket herunter.
        """
        file_path = request.query_params.get('file_path')

        if not file_path:
            return Response({"error": "Dateipfad ist erforderlich."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            content = self.bucket_manager.download_blob(file_path, t="str")
            return Response({"content": content}, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_404_NOT_FOUND)
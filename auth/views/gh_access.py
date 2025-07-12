import os

from rest_framework.views import APIView
from rest_framework.response import Response


class GHAccess(APIView):
    """
    GET /api/access/?key=123&type=head
    """
    def get(self, request):
        access_key = request.data.get("key")
        req_type = request.data.get("type")

        # TODO Vergleich mit sicher gespeicherten KEY
        """if access_key != settings.SERVER_ACCESS_KEY or req_type != "head":
            return Response({"detail": "Unauthorized."}, status=status.HTTP_403_FORBIDDEN)
        """
        if req_type == "head":
            repos = [
                "_ray_server",
                "_utils"
            ]
        elif req_type == "qfn":
            repos = [
                "_qfn",
                "_google_database",
                "_utils"
            ]


        # Erfolgreiche Antwort
        gh_info = {
            "user": os.environ.get("GH_USER"), #todo use comp gh account
            "worker_repo": os.environ.get("GH_RAY_SERVER_REPO"),
            "token": os.environ.get("GH_TOKEN")
        }
        return Response(gh_info)

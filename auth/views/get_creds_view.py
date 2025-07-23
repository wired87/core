import json
import os

from rest_framework.views import APIView
from rest_framework.response import Response



class GetCredsView(APIView):
    """
    GET /api/access/?key=123&type=head
    """
    def get(self, request):
        contact_id = request.data.get("contact_id")
        access_key = request.data.get("key")
        req_types = request.data.get("types")

        # TODO Vergleich mit sicher gespeicherten KEY
        """if access_key != settings.SERVER_ACCESS_KEY or req_type != "head":
            return Response({"detail": "Unauthorized."}, status=status.HTTP_403_FORBIDDEN)
        """
        creds = {}

        if "fb_creds" in req_types:
            local_fb_creds_path = "_google/g_auth/firebase_creds.json"
            creds["fb_creds"] = json.dumps(open(local_fb_creds_path, "r"))

        if "g_creds" in req_types:
            local_fb_creds_path = "_google/g_auth/aixr-401704-59fb7f12485c.json"
            creds["g_creds"] = json.dumps(open(local_fb_creds_path, "r"))

        if "gh" in req_types:
            # Erfolgreiche Antwort
            creds["gh"] = {
                "user": os.environ.get("GH_USER"),  # todo use comp gh account
                "worker_repo": os.environ.get("GH_RAY_SERVER_REPO"),
                "token": os.environ.get("GH_TOKEN")
            }

        return Response(creds)

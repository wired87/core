

from django.http import JsonResponse
from rest_framework import serializers
from rest_framework.views import APIView

from bm.settings import TEST_USER_ID
from workflows.create_upsert_data_def import create_process

LOAD_GRAPHP=r"C:\Users\wired\OneDrive\Desktop\Projects\Brainmaster\utils\simulator\local_graph"

class S(serializers.Serializer):
    user_id = serializers.CharField(
        required=False,
        default=TEST_USER_ID
    )
    env_id = serializers.CharField(
        help_text="ID of ENV that should be run",
        default="env_rajtigesomnlhfyqzbvx"
    )


class CreateWorldView(APIView):
    serializer_class = S
    testing = True

    def post(self, request):
        """
        Entry is alltimes 2 nodes with a edge connection
        """
        print("Create World request recieved")
        data = request.data
        env_cfg = data.get("world_cfg")


        user_id = data.get("user_id")
        env_id = data.get("env_id")

        print("Payload unpacked")

        create_process(
            user_id,
            env_cfg,
            env_id
        )

        print("Creation proces finished")
        return JsonResponse({"success": True, "env_id": env_id})










"""return FileResponse(
creator.bz_content,
as_attachment=True,
filename=f"{env_id}.zip",
content_type="application/zip"
)
"""

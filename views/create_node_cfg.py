import os

import dotenv
from django.http import JsonResponse
from rest_framework import serializers
from rest_framework.views import APIView
dotenv.load_dotenv()
from bm.settings import TEST_USER_ID
from fb_core.real_time_database import FBRTDBMgr
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from utils.g_utils import DBManager

from utils.graph.local_graph_utils import GUtils

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
    ncfg = serializers.JSONField(
        help_text="node cfg",
    )


class CreateNodeCfgdView(APIView):
    serializer_class = S
    testing = True

    def post(self, request):
        """
        Entry is alltimes 2 nodes with a edge connection
        """
        print("Create World request recieved")

        data = request.data
        ncfg = data.get("ncfg")
        user_id = data.get("user_id")
        env_id = data.get("env_id")

        print("Payload unpacked")

        self.g = GUtils(
            nx_only=True,
            g_from_path=None,
            user_id=user_id,
            enable_data_store=True
        )
        db_manager = FBRTDBMgr()
        db_manager.set_root_ref(f"users/{user_id}/env/{env_id}",)


        if "all" in nfcg:
            initial_data = db_manager._fetch_g_data()

            # Build a G from init data and load in self.g
            self.g.build_G_from_data(initial_data)

        for nid, attrs in self.g.G.nodes(data=True):
            if attrs.get("type") in ALL_SUBS:
                db_manager.upsert_data(
                    path=ncfg_path,
                    data=ncfg,
                )
        self.upsert_ncfg(
            nid,
            user_id,
            env_id,
            ncfg=ncfg,
        )
        print("Creation proces finished")
        return JsonResponse({"success": True})

    def upsert_ncfg(self, nid, user_id, env_id, ncfg):
        """
        Upsert created ENV to DB
        """
        if not user_id or len(user_id.strip()) == 0:
            user_id = TEST_USER_ID
        print("ENV ID", env_id)
        print("USER ID", user_id)
        database = f"users/{user_id}/env/{env_id}"

        ncfg_path = f"{database}/cfg/"

        instance = os.environ.get("FIREBASE_RTDB")
        db_manager = DBManager(
            table_name=None,
            instance=instance,  # set root of db
            database=database,  # spec user spec entry (like table)
            user_id=user_id,
            upload_to=["fb"]
        )

        print("Create Empty Cfg Dir")

        # UPSERT ENV CFG -> ALREADY SAVED IN NODE
        print("ENV upserted")

    def upsert_metadata(self):
        metadata_struct = {}
        nodes = [
            (nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") in [
                *ALL_SUBS, "PIXEL", "ENV"
            ]
        ]

        for nid, attrs in nodes:
            data = {
                "id": nid,
                "status": {
                    "state": "INACTIVE",
                    "info": "none"
                },
                "messages_sent": 0,
                "messages_received": 0
            }
            metadata_struct[nid] = data
        return metadata_struct






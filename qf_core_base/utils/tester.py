import asyncio

from app_utils import ENV_ID, USER_ID
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from qf_core_base.qf_utils.sim_core import SimCore
from _ray_core.ray_validator import RayValidator
from qf_core_base.utils.graph_builder import GraphBuilder
from utils.graph.local_graph_utils import GUtils


class SimTester:
    def __init__(self, host):
        self.sim = SimCore()
        self.env = ENV_ID

        if host is None:
            self.g=GUtils(user_id=USER_ID)
        else:
            self.g=None

        self.ray_validator = RayValidator(
            host=host, g=self.g
        )

        self.gbuilder = GraphBuilder(
            user_id=USER_ID,
            g=self.g,
            ray_validator=self.ray_validator
        )

    def test_run(self):
        """
        Get data buidl G
        Runn demo sim
        upsert results
        :return:
        """
        asyncio.run(self.gbuilder.main(
            env_ids=[ENV_ID],
            build_frontend_data = False,
            include_metadata=False,
            reset_g_after=False
        ))
        print("self.g.G", self.g.G)


    def build_env(self):
        """
        Starts either
        :return:
        """
        for nid, attrs in self.g.G.nodes(data=True):
            node_type = attrs.get("type").upper()
            if node_type in ALL_SUBS:
                node_attrs = {
                    "id": nid,
                    **{
                        k: v
                        for k, v in attrs.copy().items()
                        if k != "id"
                    }
                }






if __name__ == "__main__":
    tester = SimTester()
    tester.test_run()

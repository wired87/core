from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER


class DemoRunner:

    def __init__(self):
        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=None,
        )
        self.demo_g_path = r"qf_sim/_database/demo/demo_g.json"

    def create(self):
        LOGGER.info(f"Load demo graph from {self.demo_g_path}")
        return self.g.load_graph(local_g_path=self.demo_g_path)


    def finisher(self):
        """
        Load saved demo data and save as zip
        """
        pass
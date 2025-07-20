import ray

from qf_core_base.qf_utils.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER


@ray.remote
class UtilsWorker:


    def __init__(self, user_id):
        self.g = GUtils(
            nx_only=False,
            G=None,
            g_from_path=None,
            user_id=user_id,
        )
        self.qf_utils = QFUtils(
            self.g,
        )

    def set_G(self, G):
        LOGGER.info(f"Set G in UtilsWorker {self.g.user_id}")
        self.g.G=G

    def get_G(self):
        return self.g

    def get_qfu(self):
        return self.qf_utils



from qf_core_base.runner.qf_updator import QFUpdator
from qf_core_base.utils.round_finisher import Finisher
from utils.graph.local_graph_utils import GUtils


class SimRunnner:

    def __init__(self, g:GUtils):
        self.updator = QFUpdator(
            g,
            env,
            self.user_id,
        )
        self.g=g

        self.max_runs = 30
        self.event_handler = Finisher()




    def run(self):
        i = 0
        while i < self.max_runs:
            self.updator.update_qfn()
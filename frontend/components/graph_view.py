import pprint
from typing import List
from django_unicorn.components import UnicornView

from qf_core_base.qf_utils.all_subs import ALL_SUBS
from qf_sim.sim_runner import SimCore
TEST_USER_ID ="rajtigesomnlhfyqzbvx"
"""


 <script type="importmap">
            {
              "imports": {
                "three": "https://cdn.jsdelivr.net/npm/three@<version>/build/three.module.js",
                "three/addons/": "https://cdn.jsdelivr.net/npm/three@<version>/examples/jsm/"
              }
            }
            </script>


"""






class GraphViewView(UnicornView):
    """
    todo goal: serializabel_... wird von jedem node anch seinem timestep aktualisiert
    """

    template_name = "unicorn/graph_view.html"
    message: str = ""
    chat_log: List[str] = []
    dots: List[dict] = []
    nodes:list = []
    serializable_node_copy = []
    edges: list[dict] =[]
    sim_cfg = dict(
        dim = 3,
        amount_nodes = 1
    )

    def mount(self):
        print("Hi Unicorn!")

    def send(self):
        print("HI!")
        if not self.message:
            return
        self.chat_log.append(f"You: {self.message}")
        self.chat_log.append(f"Bot: Echo: {self.message}")
        self.message = ""

    """def get_nodes(self):
        if self.test is not None:
            return [attrs for nid, attrs in self.test.g.G.nodes(data=True)]
        else:
            return []
    """

    def _set_node_copy(self, runner):
        print("_set_node_copy")
        self.serializable_node_copy = runner.g.get_node_pos()
        print("serializable_node_copy set:")
        #pprint.pp(serializable_node_copy)
        #self.serializable_node_copy[nid]["state"] = attrs.get("state")

    def _set_edge_copy(self, runner):
        edges=runner.g.get_edges_src_trgt_pos()
        self.edges = edges

    def run_sim(self):
        print("RUN_SIM")
        runner = SimCore(
            user_id=TEST_USER_ID,
            env_id=f"env_bare_{TEST_USER_ID}",
            visualize=False,
            demo=True
        )

        runner.create(
                components={
                    "qf": {
                        "shape": "rect",
                        "dim": [2 for _ in range(3)]
                    }
                },
            )
        self._set_node_copy(runner)
        self._set_edge_copy(runner)







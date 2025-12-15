import ray

from app_utils import USER_ID
from cluster_nodes.cluster_utils.local_datastore import DataStore
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from utils.graph.local_graph_utils import GUtils


@ray.remote #(max_concurrency=1000)
class EdgeNode:

    """
    Operates at Pixel level
    Calcualtes coupling strenths in a concurrent loop.
    FieldWorkerNodes pick the current edge-attrs(coupling_strengths,...)
    if needed and create an edge to it

    Edges über PX grenzen
    """


    def __init__(self, edges, host, db_root, nid, ntype, px_id=None):
        self.edges:dict=edges
        self.datastore = DataStore(
            host, db_root, nid, ntype
        )
        self.px_id=px_id
        self.host=host

        self.g = GUtils(user_id=USER_ID)
        self.set_g()

        self.coupling_struct = [
            #(("type1", "type2"), self.example)
        ]

    def example(self): print("hi")

    def set_g(self):
        self.g.G = ray.get(
                self.host["UTILS_WORKER"].call.remote(
                    method_name="get_graph"
                )
            )


    async def update(self, updated_attrs):
        """receive changes and re calculate coupling for all neighbors """
        nid = updated_attrs["id"]
        node = self.g.get_node(nid)
        ntype = node.get("type")
        neighbors = self.g.get_neighbor_list(
            node=nid,
            target_type=ALL_SUBS,
        )
        if neighbors and len(neighbors):
            # recalc coupling for all
            for n in neighbors:
                # validate coupling partner
                nntype = n.get("type")
                nnid = n.get("id")
                for item in self.coupling_struct:
                    node_types = item[0]
                    if ntype in node_types and nntype in node_types:
                        coupling_term = item[1]()

                        # save updated coupling
                        self.g.G.edges[nid, nnid].update(
                            {
                                "coupling_term": coupling_term
                            }
                        )






"""
    def build_edge_G(self):
        if self.px_id is None:
            #
            pass

        edges = ray.get(
            self.host["UTILS_WORKER"].get_edges.remote()
        )

        for eid, eattrs in edges.items():
            rel = eattrs.get("rel")
            node_ids = eattrs.split(f"_{rel}_")
            src = node_ids[0]
            trgt = node_ids[1]

            # Add Edge-Node
            self.g.add_node(
                attrs={"type": "edge", **eattrs}
            )

            # Add neighbor nodes (virtual)
            self.g.add_node(
                attrs={"id": src, "type": eattrs.get("src_layer")}
            )

            self.g.add_node(
                attrs={"id": trgt, "type": eattrs.get("trgt_layer")}
            )

"""
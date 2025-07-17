import asyncio
import pprint
from concurrent.futures.thread import ThreadPoolExecutor

import ray

from cluster_nodes.updator.node import FieldWorkerNode
from qf_core_base.fermion import FERM_PARAMS
from qf_core_base.g import GAUGE_FIELDS
from qf_core_base.qf_utils.all_subs import ALL_SUBS
from qf_core_base.qf_utils.qf_utils import QFUtils
from utils.logger import LOGGER


class ClusterCreator:
    """
    Gets called inside of each qfn docker
    """

    def __init__(self, qfn_id, g, env, database, host, external_vm, session_space):
        self.g=g
        self.qf_utils = QFUtils(g)
        self.env=env
        self.database = database,
        self.host = host
        self.qfn_id=qfn_id
        self.external_vm = external_vm
        # Firebase endpoint for session data
        self.session_space = session_space


    def load_ray_remotes(self):
        """
        Called on each start after building the G
        """
        # Call in init world process befro update
        LOGGER.info("build env")
        for nid, attrs in [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") == "QFN"]:
            if nid == self.qfn_id:
                all_sub_fields = self.qf_utils.get_all_node_sub_fields(nid)
                #LOGGER.info(f"ALL EXTRACTED SUBS: {all_sub_fields}")
                # Loop all fields
                for field in all_sub_fields:
                    #LOGGER.info(f">>>field: {field}")
                    self.launch_remotes_parallel(fields=field)

    def launch_remotes_parallel(self, fields, parallel=4):
        actors = []
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(self.start_remote, nid, field) for nid, field in fields]
            for f in futures:
                actors.append(f.result())
        return actors

    def start_remote(self, nid, attrs):
        LOGGER.info(f"start_remote {nid}")

        # Get neighbors
        neighbor_types = [*ALL_SUBS, "ENV"]
        final_neighbor_struct = {}
        neighbors = self.g.get_neighbor_list(nid, neighbor_types)
        for nid, attrs in neighbors:
            final_neighbor_struct[nid] = attrs

        ref = FieldWorkerNode.options(name=nid).remote(
            self.g.G,
            attrs,
            self.env,
            self.g.user_id,
            self.database,
            self.host,
            self.external_vm,
            self.session_space,
            admin=False
        )
        self.g.G.nodes[nid]["ref"] = ref


    def extract_fields(self):
        # dieser approach ist ausberer und eh auf qfn zurückverfolgbar

        all_ferms = [k.upper() for k, v in FERM_PARAMS.items()]
        all_gs = [k.upper() for k, v in GAUGE_FIELDS.items()]
        phi = "PHI"

        ferm_params_grouped = {}
        g_params_grouped = {}
        for ferm in all_ferms:
            ferm_params_grouped[ferm.upper()] = self.add_field_to_node(ferm)
       #print("Ferms grouped")
        for g in all_gs:
            g_params_grouped[g.upper()] = self.add_field_to_node(g)
       #print("Gs grouped")
        phi = self.add_field_to_node(phi)
       #print("Higgs grouped ->")
       #print("Process finished")
        return ferm_params_grouped, g_params_grouped, phi


    def add_field_to_node(self, field):
       #print("Adding fields to", field)
        fields = []
        for nid, attrs in [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") == "QFN"]:
            ntype=attrs.get("type")
            if ntype == field:
                fields.append((nid, attrs))
        return fields




    def _handshake(self):
        """
        Request all nodes states in the _qfn_cluster_node
        """
        LOGGER.info("Check initialization state of all Nodes")
        _try = 0
        not_ready={}

        while _try != 5:
            results = asyncio.gather(*[
                ray.get(
                    attrs["ref"].receiver.receive.remote(
                        data={
                            "type": "status",
                            "data": self.env.get("id"),
                        }
                    )
                    for nid, attrs in self.g.G.nodes(data=True)
                    if attrs.get("type") in ALL_SUBS
                )
            ])

            # Check status
            for nid, status in results:
                if status == "standby":
                    continue
                not_ready[nid] = status

            if len([k for k in not_ready.keys()]) == 0:
                LOGGER.info("All nodes initialized, waiting for action")
                return True

        if len([k for k in not_ready.keys()]) > 0:
            LOGGER.info(f"Following nodes Could not be initialized:")
            pprint.pp(not_ready)
            # todo autonomous agent intervention #
            return not_ready


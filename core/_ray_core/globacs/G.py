import asyncio
import os

import networkx as nx
import ray
from ray import get_actor, ObjectRef

from ray.util.state.common import ActorState, WorkerState

from core._ray_core.base._ray_utils import RayUtils
from core._ray_core.base.base import BaseActor
from core._ray_core.globacs.state_handler.main import StateHandler
from core.app_utils import FB_DB_ROOT, SESSION_ID, USER_ID, ARSENAL_PATH
from qf_utils.all_subs import ALL_SUBS
from core.qf_utils.qf_utils import QFUtils

from utils.graph.local_graph_utils import GUtils

@ray.remote(
    num_cpus=.5,
    num_gpus=0,
)
class UtilsWorker(
    GUtils,
    BaseActor,
):

    """

    ToDo do not return whole g, create wrapper to return just admin_data needed

    """

    def __init__(self, world_cfg):
        GUtils.__init__(self)
        BaseActor.__init__(self)
        self.env_cfg = None
        self.state_handler = None

        self.qfu = QFUtils(
            self.G,
        )

        self.database = FB_DB_ROOT
        self.session_id = SESSION_ID

        self.world_cfg=world_cfg
        self.rayu = RayUtils()



    def update_ndata(self, data:ObjectRef):
        data: list[dict] = ray.get(data)
        for attrs in data:
            self.update_node(attrs)



    def get_interactive_neighbors(self):
        print("start get_interactive_neighbors")
        coupling_map = {
            field: []
            for field in ALL_SUBS
        }
        coupling_schema = [
            self.qfu.gauge_to_gauge_couplings,
            self.qfu.fermion_to_gauge_couplings,
        ]
        for item in coupling_schema:
            for field, partners in item.items():
                field = field.upper()
                if field in coupling_map:
                    #
                    coupling_map[field].extend(
                        [
                            p.upper()
                            for p in partners
                        ]
                    )

        # extend with higgs
        for field_type, partners in coupling_map.items():
            if field_type not in ["GLUON", "PHOTON"]:
                partners.append("higgs")
        print("Finished extraction interactive neihgbors ")
        ref = ray.put(coupling_map)
        return ref


    def interactionG(self):
        print("get_data_state_G")
        state_G = GUtils(
            G=nx.Graph()
        )
        try:
            for nid, node in self.G.nodes(data=True):
                if node.get("type") in ["MODULE", *os.listdir(ARSENAL_PATH)]:
                    if "nid" not in node:
                        node["nid"] = nid
                    state_G.add_node(node)
            for src, trgt, edge in self.G.edges(data=True):
                state_G.add_edge(src, trgt, edge)
            return state_G.G
        except Exception as e:
            print("Err get_data_state_G", e)



    def handle_initialized(self, host):
        try:
            self.host.update(host)
            self.state_handler = StateHandler(host=host)

            # build directly in G
            
            print(f"Received host: {host}")
        except Exception as e:
            print(f"Error updating host: {e}")


    def get_nodes_each_type(self, ref_ntypes:list[str]):
        all_ntypes = {}
        for nid, attrs in self.G.nodes(data=True):
            ntype = attrs.get("type")
            if ntype not in all_ntypes and ntype in ref_ntypes:
                all_ntypes[ntype] = attrs
        print("All nodes etracted")
        ref = ray.put(all_ntypes)
        return ref


    def get_all_edges(self, just_id):
        obj_ref = ray.put(self.get_edges(just_id=just_id))
        return obj_ref

    def get_edges(self, nid):
        return self.G.get_edge_data(nid)

    def monitor_init_state(self, helper_types:list[str]):
        # todo improve error handling ->
        #  if actor is dead: send db ->activate ErrorHandler ->
        # LIST ALL NODES
        try:
            nodes_list:list[tuple] = self.g.get_nodes(
                filter_key="type",
                filter_value=[
                    "PIXEL",
                    *ALL_SUBS,
                    *helper_types
                ]
            )

            len_keys = len(nodes_list)
            result = False

            while result is False:
                result = self.state_handler.monitor_state()
            print("INITIAL MONITORING FINISHED")
        except Exception as e:
            print(f"Err monitor_init_state: {e} ")


    def get_field_worker_ids(self) -> list[str]:
        print("get all node ids")
        all_nodes = self.g.get_nodes(
            just_id=True,
            filter_key="type",
            filter_value=ALL_SUBS
        )
        return all_nodes


    def ping(self):
        return True


    def get_all_refs(self, just_subs=False, as_dict=True, from_id_list=None):
        """
        :return: id: ref
        """

        if as_dict is True:
            refs = {}
        else:
            refs = []

        refs_of_interest = ALL_SUBS
        """
        if just_subs is False:
            refs_of_interest.append("PIXEL")
        """

        for nid, attrs in self.G.nodes(data=True):
            ntype = attrs.get("type")
            if from_id_list is not None:
                if nid in from_id_list:
                    ref = attrs["ref"]
                    refs[nid] = ref

            else:
                if ntype.upper() in refs_of_interest:
                    ref = attrs["ref"]
                    if as_dict is True:
                        refs[nid] = ref
                    else:
                        refs.append(ref)

        if as_dict is True:
            l = len(refs.keys())
        else:
            l = len(refs)

        print(f"Refs extracted: {l}")
        return refs


    async def set_workers_inactive(self):
        all_workers:dict = ray.get(get_actor(name="UTILS_WORKER").get_all_refs.remote())
        for name, ref in all_workers.items():
            ray.kill(ref)
            print(f"Worker {name} killed")

        # Upsdrt name
        await asyncio.gather(*[
            get_actor(name="DB_WORKER").iter_upsert.remote(
                attrs={
                    "status": {
                        "state": "INACTIVE",
                        "info": "null"
                    }
                },
                metadata_path=f"{self.database}/metadata/{worker.name}/"
            )
            for name, worker in all_workers.items()
        ])


    def get_ray_node_infos(self, id_map=None):
        return self.rayu.get_ray_node_infos(id_map)



    def get_logging_urls(self) -> dict:
        path_struct = {}
        for nid, attrs in self.get_ray_node_infos().items():
            worker:WorkerState = attrs.get("worker")
            actor:ActorState = attrs.get("actor")

            worker_id = worker.get("worker_id")
            job_id = worker.get("job_id", actor.get("job_id"))
            pid = worker.get("pid", actor.get("pid"))

            file_name_body = f"worker-{worker_id}-{job_id}-{pid}"

            path_struct[nid] = {
                "out": f"{file_name_body}.out",
                "err": f"{file_name_body}.err",
                "fallback": f"worker-{worker_id}-ffffffff-{pid}"
            }
        #print(f"Extract worker ids form {len(list(path_struct.keys()))} workers")
        return path_struct

    def get_npm(
            self,
            node_id:str,
            self_attrs:dict,
            all_pixel_nodes,
            env:dict
    ):
        try:
            d = env["d"]
            print("d", d)

            npm = self.qfu.npm(
                node_id,
                self_attrs,
                all_pixel_nodes
            )
            print(f"NPM for {node_id} extracted: {npm}")
            return npm
        except Exception as e:
            print(f"Error in get_npm: {e}")



    def get_all_subs_list(self, check_key="type", datastore=False, just_attrs=False, just_id=False, sort_for_types=False, sort_index_key="entry_index"):
        args = {k:v for k,v in locals().items() if k not in ["self"]}
        #print("args:", args)
        return self.qfu.get_all_subs_list(**args)


    def all_node_ids(self):
        return self.id_map

    def update_edge(self, src, trgt, attrs, rels=None):
        self.update_edge(
            src, trgt, attrs, rels
        )

    def get_neighbor(self, nid, single, *args, **kwargs ):
        if single is True:
            return self.get_single_neighbor_nx(nid, target_type=kwargs.get("trgt_type"))
        else:
            return self.get_neighbor_list_rel(nid, trgt_rel=kwargs.get("trgt_rel"))

    
    def set_G(self, G):
        print(f"Set G in UtilsWorker {USER_ID}")
        self.G=G

    def get_data_state_G(self):
        print("get_data_state_G")
        state_G = GUtils(
            G=nx.Graph()
        )
        try:
            for nid, node in self.G.nodes(data=True):
                if node.get("type") not in ["ACTOR", "CLASS_INSTANCE"]:
                    if "nid" not in node:
                        node["nid"] = nid
                    state_G.add_node(node)
            for src, trgt, edge in self.G.edges(data=True):
                state_G.add_edge(src, trgt, edge)
            return state_G.G
        except Exception as e:
            print("Err get_data_state_G", e)


    def get_G(self):
        ref = ray.put(self.G)
        return ref

    def get_qfu(self):
        return self.qfu


    def set_node_key(self, nid, key, value):
        print(f"Set key {key} = {value} -> G")
        self.G.nodes[nid][key] = value

    
    def get_all_node_sub_fields(self, nid):
        return self.qfu.get_all_node_sub_fields(
            nid=nid
        )

    def set_node(self, attrs, nid=None):
        if nid is not None and "id" not in attrs:
            attrs.update({"id": nid})

        self.add_node(
            attrs
        )

    def set_edge(self, src, trgt, attrs):
        self.add_edge(
            src=src,
            trt=trgt,
            attrs=attrs
        )

    def get_node(self, nid):
        try:
            return self.G.nodes[nid]
        except Exception as e:
            print(f"Node {nid} not found: {e}")
        return {}



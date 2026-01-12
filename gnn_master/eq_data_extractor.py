import ray
from ray import get_actor

from _ray_core.base.base import BaseActor
from _ray_core.globacs.state_handler.main import StateHandler

from core.app_utils import ENV_ID
from gnn_master.edge_utils import DataUtils
from qf_utils.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils


class NodeDataProcessor(
    DataUtils,
    BaseActor,
    StateHandler,
):

    """
    process all admin_data send inc generated pattern in batch to gpu
    """

    def __init__(
            self,
            env,
            index,
            nid,
            ntype,
            amount_nodes,
            mid,
            arsenal_struct,
            module_index,
    ):
        BaseActor.__init__(self)
        StateHandler.__init__(self)

        self.time = 0
        self.arsenal_struct=arsenal_struct

        self.amount_nodes=amount_nodes
        self.module_index=module_index
        self.g = GUtils(
            G=self.get_G()
        )
        self.qfu = QFUtils(g=self.g)

        self.updator_name = "UPDATOR"

        DataUtils.__init__(
            self,
            self.g,
            ntype,
            amount_nodes,
        )

        self.updator = get_actor("UPDATOR")
        self.guard = get_actor("GUARD")
        self.relay = get_actor(f"RELAY")

        self.nid = nid
        self.ntype = ntype
        self.index = index

        self.ferm = None
        self.edge_data = None
        self.utils_keys = None
        self.qfu = QFUtils(None)
        self.const_keys: list[dict] or None = None
        self.updator_initialized = False
        self.mid=mid
        self.item_index: int = 0
        self.utils_struct: dict[str, dict] = {}
        self.env = env
        self.parent = self.qfu.get_parent(ntype)

        self.current_eq_item = {}
        self.keys_map = []


    def set_axis(self,data):
        """
        Determines the vmap axis for each parameter in the admin_data bundle.
        - Use axis 0 for array-like admin_data (map over it).
        - Use None for scalar admin_data (broadcast it).
        """
        self.axis_def = tuple(
            0
            if not isinstance(param, (int, float))
            else None
            for param in data
        )


    def set_e_pattern(self, nodes):
        """
        # calc this on cpu
        method ensures we calc just points where time is
        existence
        """

        # todo set e param map
        energy_path_patterns = []
        for i, module in enumerate(nodes):
            for j, list_of_fields in enumerate(module):
                for param_index in list_of_fields:
                    # param_inde: int
                    energy_path_patterns.append([i, j, param_index])
        return energy_path_patterns



    def build_processor_module(
            self
    ):
        # generate admin_data
        data:dict = self.qfu.batch_field_single(
            ntype=self.ntype,
            amount_nodes=1, # create xample admin_data
            dim=self.fu.dim,
        )

        # alle receivers an relay linken (automatisch erkennen)
        self.set_axis(data)

        energy_index:int = list(data.keys()).index("energy")
        # UPDATE FIELD TYPE NODE
        self.g.update_node(
            dict(
                nid=self.nid,
                data=data,
                axis_def=self.axis_def,
                edges=self.edges,
                keys=self.keys_map,
                module_index=self.module_index,
                type="FIELD",
            )
        )

        self.g.add_edge(
            src=self.mid,
            trt=self.nid,
            attrs={
                "rel": "has_field",
                "src_layer": "MODULE",
                "trgt_layer": "FIELD",
            }
        )

        print(f"FIELD {self.nid} created successfully")



    def reload(
            self,
            data,
            return_index,
    ):
        print(f"{self.ntype} {self.item_index}/{len(self.arsenal_struct)} eqs processed")

        # BATTERY EMPTY?
        if self.item_index == len(self.arsenal_struct) - 1:
            self.finalize_iter(data)

        else:
            print("RELOAD ARSENAL")
            self.item_index += 1
            self.current_eq_item = self.eq_struct[self.item_index]
            print("current_eq_item index updated:", self.item_index)
            self.prepare_pattern_injection()


    def inject_pattern_node(self):
        """
        Cmd comes from Guard
        """
        print("================= START DATA PROCESS =================")
        param_map = self.current_eq_item[0]
        print(f"d_map {self.ntype}", param_map)

        callable = self.current_eq_item[1]
        print(f"callable {self.ntype}", callable)

        return_index = self.current_eq_item[2]
        print(f"return_index {self.ntype}", return_index)

        axis_def = tuple(self.current_eq_item[3])
        print(f"axis_def {self.ntype}", axis_def)

        self.print_payload(param_map)

        # inject_pattern
        ray.get_actor(
            name=self.updator_name
        ).main.remote(
            d_map=param_map,
            data_extractor_name=self.nid,
            axis_def=axis_def,
            runnable=callable,
            finish=self.item_index == len(self.arsenal_struct)-1,
            return_index=return_index,
            ctl_ntype=self.ntype
        )

        self.rest_eqs = len(self.arsenal_struct) - self.item_index

        print(f"arsenal shot - {self.rest_eqs} bullets left")

    def update_time(self):
        for attrs in self.attrs_struct:
            # update time
            attrs["tid"] += 1
        print("Updated time")


    def finalize_iter(self, data):
        print(f"Finalize iter")
        # update time
        self.update_time()

        self.upsert_bq()

        if self.time >= self.world_cfg["sim_time"]:
            ray.get_actor(
                name="RELAY"
            ).sim_finisher.remote(
                gid=self.parent,
            )

    def get_runnable_pattern(self, method_map:list[str], module_index):
        for p in method_map:
            param_node = self.g.G.nodes[p]





    def reset(self):
        # reset #########################
        self.edge_data = None
        self.ferm = None
        self.utils_keys = None
        self.attrs_struct: list[dict] = None
        self.global_utils = None



    ####################
    # DATA UPSERT

    def upsert_bq(self):
        """
        create copy -> extend id with time (make unique)
        extend bq table with entire admin_data structs.
        goal: full ds fits into single table
        """
        print("upsert_bq")
        keys_to_exclude = [
            *self.sum_util_keys,
            *self.const_keys,
            "type",
            "parent",
        ]

        bq_payload = []
        for attrs in self.attrs_struct:
            item_copy = attrs.copy()
            # bq format -> ensure all admin_data fits into single table
            item_copy["nid"] = f"{attrs['nid']}_{attrs['tid']}"

            # add filtered item
            bq_payload.append(
                {
                    k: v
                    for k, v in item_copy.items()
                    if k not in keys_to_exclude
                }
            )

        # UPSERT BATCH
        ray.get(
            ray.get_actor(
                name="BQ_WORKER"
            ).insert.remote(
                rows=bq_payload,
                table=ENV_ID
            )
        )
        print("ROUND BATCH UPSERTED")


@ray.remote(num_cpus=.2)
class NodeProcessorWorker(NodeDataProcessor):
    def __init__(
            self,
            env,
            index,
            nid,
            ntype,
            amount_nodes,
            mid,
            arsenal_struct,
            module_index,
    ):
        NodeDataProcessor.__init__(
            self,
            env,
            index,
            nid,
            ntype,
            amount_nodes,
            mid,
            arsenal_struct,
            module_index,
        )



"""
    def set_edges_fields(
            self,
            fields,
    ):
        # Fix: Initialize with known types instead of m[0] which causes KeyError
        interactants = {
            "FERMION": [],
            "GAUGE": [],
            "HIGGS": [],
        }

        # Create nid to index map for faster lookup
        ntype_map = {
            f["type"]: i
            for i, f in enumerate(fields)
        }

        # GT INTERACTANTS OF NEIGHBORS
        for attrs in list(self.g.get_neighbor_list_rel(
            node=self.nid,
            trgt_rel="has_finteractant",
        ).values()):
            parent = self.qfu.get_parent(attrs.get("type"))

            # Fix: Use nid to find index
            nntype = attrs["nid"]
            if nntype in ntype_map:
                if parent not in interactants:
                    interactants[parent] = []

                # classify findex to nmap
                interactants[parent].append(
                    ntype_map[nntype]
                )

            else:
                print(f"Neighbor {nntype} not found in fields list")

        print("Interactants field set successfully!")
        self.edges = list(interactants.values())


    def await_order(self):
        print(f"{self.ntype} awaiting loading")
        index = ray.get(
            get_actor(
                self.updator_name
            ).get_index.remote())

        while self.index != index:
            # update index
            try:
                index = int(ray.get(get_actor(
                    self.updator_name
                ).get_index.remote()))
                print(f"sleep 5s {self.ntype} {self.index}:{index}")
            except Exception as e:
                print(f"Err await_order: {e}")
            time.sleep(5)
        print(f"{self.ntype} ({self.index}) index received: {index} -> ready to fire")

"""
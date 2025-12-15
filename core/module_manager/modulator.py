import os
import time
from typing import Any

import ray
from ray import get_actor

from _ray_core.globacs.state_handler.main import StateHandler
from app_utils import ENVC, DIM
from code_manipulation.eq_extractor import EqExtractor
from gnn.eq_data_extractor import NodeDataProcessor
from gnn.mod import GNNModuleBase
from module_manager.module_loader import ModuleLoader
from qf_utils.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils


@ray.remote
class Modulator(
    StateHandler,
    ModuleLoader,
    EqExtractor,
):
    def __init__(
            self,
            g,
            mid:str,
            qfu:QFUtils,
            module_index,
            amount_nodes
    ):
        self.id = mid
        self.g:GUtils = g
        self.qfu = qfu

        StateHandler.__init__(self)
        EqExtractor.__init__(self, g)

        self.amount_nodes=amount_nodes
        self.attrs = self.g.G.nodes[mid]
        self.await_alive(
            ["UTILS_WORKER"]
        )
        self.module_index=module_index


    def module_conversion_process(self):
        attrs = self.g.G.nodes[self.id]
        path = attrs["path"]
        if os.path.isdir(path):
            # file_module
            # load files
            # param map extract
            # -> include G with keys and type
            # struct!!!!
            # align key params existing fields (extension?)
            # link
            # todo
            pass

        self.fields = self.g.get_neighbor_list_rel(
            node=self.id,
            trgt_rel="has_finteractant",
            as_dict=True
        )

        # init here because  if isdir fields, params mus be def
        ModuleLoader.__init__(
            self,
            G=self.g.G,
            nid=self.id,
            fields=self.fields,
        )

        # create workers for each field
        self.create_field_workers(self.fields)

        # create params
        self.create_field_stack()

        # convert
        # build
        # convert G
        # return callable
        self.load_local_module_codebase()

        # code -> G
        self.create_code_G(mid=self.id)

        # G -> sorted runnables -> add inde to node
        self.set_arsenal_struct()

        # create nnx.Modules
        self.create_modules()

        # map param -> operator pytree ->
        for item in self.arsenal_struct:
            self.process(
                item["code"],
                parent_id=self.id,
                module_id=item["nid"],
            )


        print("MODULE CONVERSION COMPLETE")
        # MESSAGE GUARD: MODULE READY
        get_actor("GUARD").handle_module_ready.remote(
            self.id,
            list(self.fields.keys()),
        )


    def create_field_stack(self):
        """

        BRING PARAMS FOR ENTIRE MODULE
        TO SEPARATED dict[SOA[SOA]]

        """
        print("create_field_stack start")
        module_param_struct = {}

        fields = self.g.get_neighbor_list(
            node=self.id,
            target_type="FIELD"
        )

        for f in fields:
            data: dict = self.qfu.batch_field_single(
                ntype=f,
                amount_nodes=self.amount_nodes,
                dim=DIM,
            )
            for k, soa in data.items():
                module_param_struct[k] = [soa]

        self.keys: list[str] = list(module_param_struct.keys())
        self.values = list(module_param_struct.values())

        self.set_field_data()
        print(f"Creation of dict[soa[soa]] for module {self.id} finished")


    def set_field_data(self):
        # add method index to node to ensure persistency in equaton handling
        self.g.update_node(
            dict(
                nid=self.id,
                keys=self.keys,
                value=self.values,
            )
        )
        print("create_modules finished")


    def create_modules(self):
        # add method index to node to ensure persistency in equaton handling
        for i, item in enumerate(self.arsenal_struct):
            self.g.update_node(
                dict(
                    nid=item["nid"],
                    method_id=i,
                    module=GNNModuleBase()
                )
            )
        print("create_modules finished")



    def set_pattern(self):
        STRUCT_SCHEMA = [
            # module index
            # findex
            # list pindex
        ]

        for f in self.fields:
            node = self.g.G.nodes[f]
            keys = node["keys"]

            # loop single eqs
            for struct in self.arsenal_struct:
                for p in struct["params"]:
                    struct_item = []
                    # param lccal?
                    if p in keys:
                        struct_item = [
                            self.module_index,
                             node["field_index"],
                            keys.index(p)
                        ]

                    elif p in ENVC:
                        struct_item = [
                            self.module_index,
                            [0], # first and single field
                            ENVC.index(p),
                        ]

                    else:
                        # param from neighbor field ->
                        # get all NEIGHBOR FIELDS
                        nfs = self.g.get_neighbor_list(
                            node=f["nid"],
                            target_type="FIELD",
                        )

                    STRUCT_SCHEMA[
                        node["field_index"]
                    ] = struct_item



    def create_grid(self):
        """
        Fields by their own create an upsert thei data by owns.
        JUST USE FOR RESTART
        """
        print("Creating grid...")
        start = time.perf_counter_ns()
        tasks = []

        for nid, attrs in self.fields.items():
            tasks.append(
                attrs["edge_ref"].build_processor_module.remote(
                )
            )
        ray.get(tasks)
        end = time.perf_counter_ns()
        print("Grid created successfully after s:", end - start)




    def set_return_des(self):
        # param: default v
        field_param_map: dict[str, Any] = self.g.G.nodes[self.id]["field_param_map"].keys()

        # create PATTERN
        k = list(field_param_map.keys())
        return_param_pattern = [
            None
            for _ in range(len(k))
        ]

        # LINKFIELD PARAM -> RETURN KEY
        for i, item in enumerate(self.arsenal_struct):
            return_key = item["return_key"]
            return_param_pattern[i]: int = field_param_map.index(return_key)

        print(f"{self.id} runnable creared")



    def create_field_workers(self, fields):
        # todo field utils -> remote
        start = time.perf_counter_ns()

        for i, attrs in enumerate(fields):
            if "ref" not in attrs:
                uname = f"DATA_EXTRACTOR_{attrs['nid'].upper()}"

                # 2. FINDEN DES ZUGEHÖRIGEN MODULE-INDEX
                # Bestimmt den Parent-Typ des Feldes (sollte der NID des Moduls sein)
                parent_ntype = self.qfu.get_parent(attrs["nid"])

                # Sucht den Index des Moduls in der sortierten Liste
                # Nutzt -1 als Marker, falls das Modul nicht gefunden wird
                ref = NodeDataProcessor.options(
                    name=uname,
                    lifetime="detached",
                ).remote(
                    #field_index=i, # todo index apply at reorder
                    ntype=attrs["nid"],
                    parent=parent_ntype,
                    nid=uname,
                    env=ENVC,
                    all_fields=[a["nid"] for a in fields],
                    mid=self.id,
                    module_index=self.attrs["module_index"],
                )
                try:
                    self.g.update_node({
                        "nid": attrs["nid"],
                        "ref": ref,
                    })

                except Exception as e:
                    print(f"Error creating DATA_EXTRACTOR for node: {e}")

        end = time.perf_counter_ns()
        print("Field Workers created successfully after s:", end - start)




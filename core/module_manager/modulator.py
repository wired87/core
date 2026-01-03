import time
from typing import Any


from code_manipulation.eq_extractor import EqExtractor
from core.module_manager.module_loader import ModuleLoader
from qf_utils.field_utils import FieldUtils
from qf_utils.qf_utils import QFUtils

from gnn_master.eq_data_extractor import NodeDataProcessor
from utils.graph.local_graph_utils import GUtils


class Modulator(
    ModuleLoader,
    EqExtractor,
):
    def __init__(
            self,
            G,
            mid:str,
            qfu:QFUtils,
            module_index:int,
    ):
        self.id = mid
        self.g:GUtils = GUtils(G=G)
        self.qfu = qfu
        self.fu = FieldUtils()
        #StateHandler.__init__(self)
        EqExtractor.__init__(self, self.g)

        self.module_index=module_index


    def module_conversion_process(self, module_index):
        try:
            attrs = self.g.G.nodes[self.id]
            sm = attrs["sm"]
            if sm is False:
                """
                prompt:
                extract the following information from th povided physics papaer.
                - the published field (elektron or other)
                - the published fields interactant 
                - parameters with example value present within the field
                - equations within the publication 
                """
                # file_module
                # load files
                # param map extract
                # -> include G with keys and type
                # struct!!!!
                # align key params existing fields (extension?)
                # link
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

            # create workers and params for each field
            self.create_field_workers(list(self.fields.keys()))

            # convert
            # build
            # convert G
            # return callable
            self.load_local_module_codebase()

            # code -> G
            self.create_code_G(mid=self.id)

            # G -> sorted runnables -> add inde to node
            self.set_arsenal_struct()
            print("self.arsenal_struct", self.arsenal_struct)


            print("MODULE CONVERSION COMPLETE")
        except Exception as e:
            print("MODULE CONVERSION FAILED:", e)



    def set_field_data(self, field):
        """
        Set example field data
        """
        print("set_field_data")
        try:

            data: dict = self.qfu.batch_field_single(
                ntype=field,
                dim=self.fu.env,
            )

            # set params for module
            keys = list(data.keys())
            values = list(data.values())
            axis_def = self.set_axis(values)

            print(f"update field node {field} ")
            self.g.update_node(
                dict(
                    nid=field,
                    keys=keys,
                    values=values,
                    axis_def=axis_def,
                )
            )
        except Exception as e:
            print("Err set_field_data:", e)
    print("create_modules finished")


    def create_modules(self):
        # add method index to node to ensure persistency in equaton handling
        for i, item in enumerate(self.arsenal_struct):
            self.g.update_node(
                dict(
                    nid=item["nid"],
                    method_id=i,
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

                    elif p in self.fu.env:
                        struct_item = [
                            self.module_index,
                            [0], # first and single field
                            self.fu.env.index(p),
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



    def create_field_workers(
            self,
            fields:list[str]
    ):
        # todo may add module based axis def
        start = time.perf_counter_ns()
        try:
            for i, fid in enumerate(fields):
                self.set_field_data(
                    field=fid,
                )
        except Exception as e:
            print("Err create_field_workers", e)
        end = time.perf_counter_ns()
        print("Field Workers created successfully after s:", end - start)

    def set_axis(self, data:list) -> tuple:
        """
        Determines the vmap axis for each parameter in the admin_data bundle.
        - Use axis 0 for array-like admin_data (map over it).
        - Use None for scalar admin_data (broadcast it).
        """
        return (
            0
            if not isinstance(
                param, (int, float)
            )
            else None
            for param in data
        )

"""    
def create_field_stack(self):
    print("create_field_stack start")
    module_param_struct = {}

    fields = self.g.get_neighbor_list(
        node=self.id,
        target_type="FIELD",
        just_ids=True,
    )

    for f in fields:
        self.set_field_data(
            f,
            module_param_struct
        )

    self.keys: list[str] = list(module_param_struct.keys())
    self.values = list(module_param_struct.values())

    # add method index to node to ensure persistency in equaton handling
    self.g.update_node(
        dict(
            nid=self.id,
            keys=self.keys,
            value=self.values,
        )
    )
"""
"""    
def create_grid(self):
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
"""
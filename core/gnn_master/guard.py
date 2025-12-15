from typing import Any

import jax
import ray
from ray import get_actor

from _god.create_world import God
from _ray_core.globacs.state_handler.main import StateHandler
from app_utils import FB_DB_ROOT, ENVC
from chain import GNNChain
from gnn import DataUtils
from gnn import GNN
from gnn import GNNModuleBase

from gnn_master.pattern_store import GStore
import jax.numpy as jnp

from module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS

class Guard(
    StateHandler,
    DataUtils,
    God,
):
    # todo answer caching
    # todo cross module param edge map
    """
    nodes -> guard: extedn data
    """

    def __init__(
        self,
        world_cfg,
        qfu,
        g,
    ):
        print("Initializing Guard...")
        super().__init__()
        God.__init__(
            self,
            g,
            qfu,
            world_cfg
        )

        self.amount_nodes = world_cfg["amount_nodes"]

        self.gpu = get_actor("UPDATOR")
        self.time = 0
        self.world_cfg = world_cfg

        self.schema_grid = [
            (i,i,i)
            for i in range(len(self.amount_nodes))
        ]

        self.qfu = qfu
        self.g = g

        self.ready_map = {
            k: False
            for k in ALL_SUBS
        }

        self.updator_name = "UPDATOR"
        self.database = FB_DB_ROOT

        # Time Series Model Init
        self.prev_state = None
        self.model_params = None
        self.fnished_modules = False

        self.fields = []
        self.store = [
            []
            for _ in range(self.amount_nodes)
        ]
        print("Guard Initialized!")


    def xtend_store(self, obj_ref, index:int):
        self.store[index] = jax.device_put(
            ray.get(obj_ref)
        )
        print("All data placed for", index)
        return


    def handle_module_ready(
            self,
            mid,
    ):
        """
        Trigger when module was created and fields initialized
        runnable alread include
        """
        try:
            self.g.G.nodes[mid]["ready"] = True
        except Exception as e:
            print("Err receive_ready", mid, e)

        # all modules are ready build?
        if self.all_nodes_ready(self.modules):
            # build the model structure
            self.build_gnn()

            # pattern extractor
            self.compile_pattern()

            # load module in GNN
            self.load_content_gnn()


    def load_content_gnn(self):
        # sort everything to the gnn
        print("load_content_gnn...")
        get_actor("GNN").main.remote(
            modules=self.pattern_arsenal
        )
        print("load_content_gnn... done")



    def main(self):
        # create MODULES
        self.create_injector()

        # create interaction standard model
        self.qfu.build_interacion_G()

        # create env
        self.create_env()

        # create modules
        self.module_creator_worker()





    def set_param_edge_pattern(self):
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        for m in modules:
            # get param (method) neighbors
            # get params
            # extract index
            module_index = m["module_index"]
            fields = self.g.get_neighbor_list(
                node=m["nid"],
                target_type="PARAM"
            )

            tasks = [
                get_actor(f).get_runnable_pattern.remote(
                )
                for f in fields
            ]
            ray.get(tasks)

        return

    def set_param_index_map(self):
        """
        Map
        """
        # create param index map
        pindex_map_fields = [
            [] for _ in range(
            len(self.fields))
        ]

        # get all modules
        fields = self.g.get_nodes(
            filter_key="type",
            filter_value="FIELD",
        )

        # todo sync finish state of all
        #  fields before
        for fid, f in fields:
            arsenal_struct = f["arsenal_struct"]
            param_map = arsenal_struct["params"]
            param_index_map = [[], []]
            for param in param_map:
                # param: str
                if param in f["keys"]:
                    # first param mapping for index
                    param_index_map[0].append(None)
                    param_index_map[1].append(
                        f["keys"].index(param)
                    )

                pindex_map_fields[
                    self.fields.index(fid)
                ] = param_index_map

                self.g.update_node(
                    dict(
                        fid,
                        param_index_map=param_index_map
                    )
                )
            print("create_param_index_map finisehd:", )
            return pindex_map_fields



    def create_injector(self):
        from injector import Injector
        self.injector = Injector.options(
            name="INJECTOR",
            lifetime="detached"
        ).remote(
            world_cfg=self.world_cfg
        )

        self.g.add_node(
            dict(
                nid="INJECTOR",
                ref=self.injector._ray_actor_id.binary().hex(),
                type="ACTOR",
            )
        )

        self.await_alive(["INJECTOR"])
        print("create_injector")



    def build_gnn(self):
        self.gnn = GNN.options(
            "GNN",
            lifetime="detached",
        ).remote(
            modules_len=len(
                self.modules
            ),
            amount_nodes=self.amount_nodes,
            glob_time=self.world_cfg["sim_time_s"],
        )






    def create_and_distribute_data(self):
        """
        Loop modules -> fields -> data(params)
        write param to nnx.Module
        """

        # create pattern
        modules:list[tuple] = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        for mid, mttrs in modules:
            # get module
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )

            for fid, fattrs in fields:
                params:dict[str, Any] = fattrs["data"]

                for i, (k, v) in enumerate(params.items()):
                    # get param module
                    param = self.g.G.nodes[k]
                    module = param["module"]

                    # add data
                    module.add_data(
                        v,
                        self.amount_nodes
                    )

        print("Params sorted to nodes")
        return


    def all_nodes_ready(self, trgt_types) -> bool:
        return all(
            attrs["ready"] is True
            for attrs in list(
                self.g.G.nodes[ntype]
                for ntype in trgt_types
            )
        )


    def module_creator_worker(self):
        # Create Lexi -> load arsenal
        ref = ModuleCreator.options(
            lifetime="detached",
            name="MODULE_CREATOR"
        ).remote()
        self.modules = ray.get(ref.main.remote())





    def compile_pattern(self):
        try:
            # GET MODULE todo filter for Guard
            modules = self.g.get_nodes(
                filter_key="type",
                filter_value="MODULE"
            )

            self.module_pattern_collector = []

            for i, module in enumerate(modules):
                mid = module["nid"]
                keys: list[str] = module["keys"]
                faxis_def: list[int or None] = module["axis_def"]


                # get module fields
                fields = self.g.get_neighbor_list(
                    node=mid,
                    target_type="FIELD",
                )

                meq = self.get_module_eqs(mid)

                # set empty param map struct struct
                modules_param_map = [
                    []
                    for _ in range(len(meq))
                ]

                for fid, fattrs in fields:
                    field_index = fattrs["field_index"]

                    for mindex, (eqid, eqattrs) in enumerate(meq):
                        # ALL PARAMS MAPPED OF METHOD
                        method_param_collector:list[
                            list[Any]
                        ] = []

                        axis_def = []

                        # get all params from
                        # todo mark order
                        params = self.g.get_neighbor_list(
                            node=eqid,
                            target_type="PARAM",
                        )

                        for pid, pattrs in params:
                            if pid in keys:
                                pindex = keys.index(pid)
                                axis_def.append(faxis_def[pindex])
                                method_param_collector.append(
                                    [
                                        i,
                                        pindex,
                                        field_index
                                    ]
                                )
                            elif pid in ENVC:
                                pindex = list(ENVC.keys()).index(pid)

                                # add axis def
                                axis_def.append(None)

                                method_param_collector.append(
                                    [
                                        0, # envc altimes module 0
                                        pindex,
                                        0 # envc single field
                                    ]
                                )
                            else:
                                # check
                                for j, module in enumerate(modules):
                                    nmkeys: list[str] = module["keys"]
                                    if pid in nmkeys:
                                        pindex = nmkeys.index(pid)

                                        # loop fields of module

                                        fneighbors = self.g.get_neighbor_list(
                                            node=fid,
                                            target_type="has_finteractant",
                                        )

                                        # get neighbors from field
                                        # mark: nfield-index represents row
                                        # of GT
                                        for nfid, nfattrs in fneighbors:
                                            nfield_index = nfattrs["field_index"]
                                            nfaxis_def: list[str] = module["axis_def"]

                                            # add axis def
                                            axis_def.append(nfaxis_def[nfield_index])

                                            # append method to
                                            method_param_collector.append(
                                                [
                                                    j,
                                                    pindex,
                                                    nfield_index
                                                ]
                                            )

                        return_index_map = [
                            i,
                            keys.index(eqattrs["return_key"]),
                            field_index,
                        ]

                        modules_param_map[mindex].append(
                            GNNModuleBase(
                                runnable=eqattrs["callable"],
                                inp_patterns=method_param_collector,
                                outp_pattern=return_index_map,
                                in_axes_def=axis_def,
                                method_id=mindex,
                            )
                        )

                        print(f"EQ pattern for {eqid} written")

                self.module_pattern_collector.append(
                    GNNChain(
                        method_modules=modules_param_map
                    )
                )
                print(f"EQ pattern for written")

            # Create GStore
            self.pattern_arsenal = GStore(
                nodes=self.store,
                edges=[], # todo get edges
                inj_pattern=[], # todo get inj pattern
                method_struct=self.module_pattern_collector
            )
            print("GStore compiled")

        except Exception as e:
            print(f"Err compile_pattern: {e}")



    def get_module_eqs(self, mid):
        # get methods for module
        meq = self.g.get_neighbor_list(
            node=mid,
            target_type="METHOD",
        )

        # bring to exec order
        meq = sorted(meq, key=lambda x: x["method_index"], reverse=True)
        return meq

@ray.remote
class GuardWorker(Guard):
    def __init__(
        self,
        world_cfg,
        qfu,
        g,
    ):
        Guard.__init__(
            self,
            world_cfg,
            qfu,
            g,
        )




import base64
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

# import ray
import ray
from ray import get_actor


from _god.create_world import God
from _ray_core.globacs.state_handler.main import StateHandler

from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
from core.app_utils import ENVC

from gnn_master.pattern_store import GStore
from module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS

class Guard(
    StateHandler,
    #DataUtils,
    God,
):
    # todo answer caching
    # todo cross module param edge map
    """
    nodes -> guard: extedn data
    """

    def __init__(
        self,
        qfu,
        g,
    ):
        print("Initializing Guard...")
        super().__init__()
        God.__init__(
            self,
            g,
            qfu,
        )

        self.world_cfg=None
        self.inj_pattern=None
        self.artifact_admin = ArtifactAdmin()

        self.gpu = get_actor("UPDATOR")
        self.time = 0

        self.qfu = qfu
        self.g = g

        self.ready_map = {
            k: False
            for k in ALL_SUBS
        }

        self.mcreator = ModuleCreator()
        self.modules = self.mcreator.load_sm()

        self.updator_name = "UPDATOR"

        self.pattern_arsenal = GStore

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

    def set_inj_pattern(
            self,
            inj_pattern:dict[
                str, dict[tuple, list[list, list]]
            ] # pos:time,val -> entire sim len captured
    ):
        self.inj_pattern = [[] for _ in range(
            len(self.schema_grid)
        )]
        for ipos, item in inj_pattern:
            self.inj_pattern[
                self.schema_grid.index(ipos)
            ]:list[list, list] = item
        print("set_inj_pattern finished")



    def set_wcfg(self, world_cfg):
        self.world_cfg = world_cfg
        self.amount_nodes = world_cfg["amount_nodes"]
        self.schema_grid = [
            (i, i, i)
            for i in range(len(self.amount_nodes))
        ]


    def xtend_store(self, obj_ref, index:int):
        self.store[index] = ray.get(obj_ref)
        print("All data placed for", index)
        return


    def handle_module_ready(
            self,
            mid,
            field_ids
    ):
        """
        Trigger when module was created and fields initialized
        runnable alread include
        """
        try:
            self.g.G.nodes[mid]["ready"] = True
        except Exception as e:
            print("Err receive_ready", mid, e)


    def deploy_sim(self, user_id, env_id):
        from workflows.deploy_sim import DeploymentHandler
        self.deployment_handler = DeploymentHandler(
            user_id
        )

        # all modules are ready build?
        finished_modules=False
        while not finished_modules:
            finished_modules = self.all_nodes_ready(self.modules)
            time.sleep(1)

        # pattern extractor -> include all fields
        self.compile_pattern()
        self.create_db()
        docker_payload = {
            "UPDATOR_PATTERN":self.module_pattern_collector,
            "DB": self.db,
            "AMOUNT_NODES": self.amount_nodes,
            "INJECTION_PATTERN": self.inj_pattern,
            "TIME": self.world_cfg,
        }

        # DEPLOY
        container_env = self.env_creator.create_env_variables(
            env_id=env_id,
            cfg=docker_payload
        )

        self.deployment_handler.create_vm(
            instance_name=env_id,
            testing=self.testing,
            image=self.artifact_admin.get_latest_image(),
            container_env=container_env
        )
        print("deploy_sim finsihed")


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

    def handle_mod_stack(self, files:list[Any]):
        """
        receive files
        write to temp
        load in ModuleCreator -> create modules
        """

        tmp = TemporaryDirectory()
        root = Path(tmp.name)

        for f in files:
            # Pflichtfelder
            fname = f["name"]
            rel_path = f.get("path", fname)
            raw = base64.b64decode(f["data"])

            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

        self.modules.append(self.mcreator.main(
            temp_path=root
        ))

        print("modules created successfully")







    def module_creator_worker(self):
        # Create Lexi -> load arsenal
        ref = ModuleCreator.options(
            lifetime="detached",
            name="MODULE_CREATOR"
        ).remote()
        self.modules = ray.get(ref.main.remote())


    def create_db(self):
        """
        collect all nodes values and stack
        """
        self.db = []

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )
        for mid, m in modules:
            mod_db = []
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )


            for fid, fattrs in fields:
                node = self.g.get_node(nid=fid)
                mod_db.append(
                    [
                        node["value"],
                        node["axis_def"]
                    ]
                )
            self.db.append(mod_db)



    def compile_pattern(self):
        def get_param(*args):
            return [
                *args,
            ]

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
                            pval = pattrs["value"]

                            #
                            if pid in keys:
                                pindex = keys.index(pid)

                                axis_def.append(faxis_def[pindex])
                                method_param_collector.append(
                                    get_param(
                                        [j,
                                        pindex,
                                        nfield_index,]
                                    )
                                )

                            # ENV ATTR
                            elif pid in ENVC:

                                pindex = list(ENVC.keys()).index(pid)
                                # add axis def
                                axis_def.append(None)

                                modules_param_map[i].append(
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
                                                get_param(
                                                    j,
                                                    pindex,
                                                    nfield_index,
                                                )
                                            )

                        return_index_map = [
                            i,
                            keys.index(eqattrs["return_key"]),
                            field_index,
                        ]

                        modules_param_map[i].append(
                            [
                                eqattrs["callable"],
                                method_param_collector,
                                return_index_map,
                                axis_def,
                                mindex,
                            ]
                        )
                        """
                        GNNModuleBase(
                            runnable=,
                            inp_patterns=method_param_collector,
                            outp_pattern=return_index_map,
                            in_axes_def=axis_def,
                            method_id=mindex,
                        )"""

                        print(f"EQ pattern for {eqid} written")

                    self.module_pattern_collector.append(
                        modules_param_map
                    )
                    """
                    GNNChain(
                            method_modules=modules_param_map
                        )
                    """
                    print(f"EQ pattern for written")
        except Exception as e:
            print(f"Err compile_pattern: {e}")

    """
    compile self.module_pattern_collector -> GNNChain
    compile modules_param_map -> GNNModuleBase
    """

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




import base64
import pprint
import time
from typing import Any

from _god.create_world import God
from core._ray_core.globacs.state_handler.main import StateHandler

from bob_builder.artifact_registry.artifact_admin import ArtifactAdmin
from core.app_utils import SCHEMA_GRID

from gnn_master.pattern_store import GStore
from core.module_manager.mcreator import ModuleCreator
from qf_utils.all_subs import ALL_SUBS
from qf_utils.field_utils import FieldUtils
from workflows.deploy_sim import DeploymentHandler

class PatternMaster:

    def __init__(self, g):
        self.g=g
        self.modules_struct=[]


    def get_empty_field_structure(self):
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        self.modules_struct = [[] for _ in range(len(modules))]

        for i, (mid, m) in enumerate(modules):
            # get module fields
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )
            field_struct = [[] for _ in range(len(fields))]

            # SET EMPTY DIMS FOR EACH FIELD
            self.modules_struct[i] = field_struct
        return self.modules_struct



class Guard(
    StateHandler,
    God,
    FieldUtils,
PatternMaster,
):
    # todo answer caching
    # todo cross module param edge map
    """
    nodes -> guard: extedn admin_data
    """

    def __init__(
        self,
        qfu,
        g,
        user_id,
    ):
        print("Initializing Guard...")
        super().__init__()
        God.__init__(
            self,
            g,
            qfu,
        )
        PatternMaster.__init__(
            self,
            g,
        )
        self.user_id=user_id
        self.deployment_handler = DeploymentHandler(
            user_id
        )
        self.world_cfg=None
        self.artifact_admin = ArtifactAdmin()

        self.time = 0

        self.qfu = qfu
        self.g = g
        self.ready_map = {
            k: False
            for k in ALL_SUBS
        }

        self.mcreator = ModuleCreator(
            self.g.G,
            self.qfu,
            self.world_cfg,
        )
        self.modules = self.mcreator.load_sm()

        self.pattern_arsenal = GStore

        # Time Series Model Init
        self.prev_state = None
        self.model_params = None
        self.fnished_modules = False

        self.fields = []

        print("Guard Initialized!")


    def main(self, env_ids:list[str]):
        """
        CREATE/COLLECT PATTERNS FOR ALL ENVS AND CREATE VM
        """

        # all modules are ready build?
        finished_modules = False
        while not finished_modules:
            print("check modules ready")
            all_modules = [nid for nid, _ in self.g.get_nodes(filter_key="type", filter_value="MODULE")]
            finished_modules = self.all_nodes_ready(all_modules)
            time.sleep(1)

        # pattern extractor -> include all fields
        # need
        # inj
        # wcfg
        self.compile_pattern()
        self.create_db()
        self.set_param_pathway()

        for env in env_ids:
            inj_pattern = self.g.get_neighbor_list(
                node=env,
                target_type="INJECTION_PATTERN",
            )

            # CREATE VM PAYLOAD
            docker_payload = {
                "UPDATOR_PATTERN": self.module_pattern_collector,
                "DB": self.db,
                "AMOUNT_NODES": self.amount_nodes,
                "INJECTION_PATTERN": inj_pattern,
                "TIME": self.world_cfg,
                "ENERGY_MAP": self.module_map_entry
            }
            print(f"DOCKER ENV VARS CREATED FOR {env}:")
            pprint.pp(docker_payload)

            # PUBLISH VM
            self.deploy_sim(
                env,
                env_var_cfg=docker_payload
            )

        print(f"Deployment of {env_ids} Finished!")


    def get_inj_pattern_data(self):
        return {
            "amount_nodes": self.amount_nodes,
            "fields": self.fields,
        }


    def get_state(self):
        return len(self.fields) and self.world_cfg


    def set_inj_pattern(
            self,
            inj_pattern:dict[
                str, dict[tuple, list[list, list]]
            ], # pos:time,val -> entire sim len captured
            env_id:str
    ):
        """
        frontend -> relay-> inj_pattern
        self.inj_pattern = [
            [pos, [[t],[e]]],
        ]
        """
        # retrieve empty struct based on all fieds and modules creatde
        inj_struct = self.get_empty_field_structure()

        for ntype, pos_inj_struct in inj_pattern.items():
            fattrs = self.g.G.nodes[ntype]
            module_index: int = fattrs["module_index"]
            field_index:int = fattrs["field_index"]
            for pos, inj_pattern in pos_inj_struct:
                inj_struct[
                    module_index
                ][
                    field_index
                ][
                    SCHEMA_GRID.index(pos)
                ] = inj_pattern

        # INJECTION_STRUCT -> G
        inj_id = f"inj_{env_id}"
        self.g.add_node(
            dict(
                nid=inj_id,
                pattern=inj_struct,
                type="INJECTION_PATTERN",
            )
        )

        # ENV -> INJECTION_STRUCT
        self.g.add_edge(
            src=env_id,
            trgt=inj_id,
            attrs=dict(
                src_layer="ENV",
                trgt_layer="INJECTION_PATTERN",
                rel="has_injection_pattern",
                # **edge_yaml_cache[edge_key],
            )
        )
        print("set_inj_pattern finished")


    def set_wcfg(self, world_cfg):
        self.world_cfg = world_cfg

        self.amount_nodes = world_cfg["amount_nodes"]

        self.schema_grid = [
            (i, i, i)
            for i in range(self.amount_nodes)
        ]

        node = {
            "nid": world_cfg["env_id"],
            "cfg": world_cfg,
            "params": self.create_env(),
            "type": "ENV",
        }

        self.g.add_node(node)
        return node




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
            self.g.G.nodes[mid]["field_keys"] = field_ids
        except Exception as e:
            print("Err receive_ready", mid, e)


    def set_param_pathway(
            self,
            trgt_key="energy"
    ) -> list[list[int]]:
        # exact same format
        print("set_param_pathway started")
        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        self.module_map_entry = [
            [] for _ in range(len(modules))
        ]

        for mid, mattrs in modules:
            # get param (method) neighbors
            # get params
            # extract index
            mindex = mattrs["module_index"]
            fields:list[tuple] = self.g.get_neighbor_list(
                node=mattrs["nid"],
                target_type="FIELD"
            )
            energy_param_map=[
                None
                for _ in range(len(fields))
            ]
            for fid, fattrs in fields:
                try:
                    energy_param_map[fattrs["field_index"]
                    ] = fattrs["field_keys"].index(
                        trgt_key
                    )
                except Exception as e:
                    print(f"Err map pathway to param:{trgt_key}: {e}")
            self.module_map_entry[mindex] = energy_param_map
        print(f"param map to {trgt_key} build")


    def deploy_sim(self, env_id, env_var_cfg):
        # DEPLOY
        container_env = self.deployment_handler.env_creator.create_env_variables(
            env_id=env_id,
            cfg=env_var_cfg
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
        Loop modules -> fields -> admin_data(params)
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
                params:dict[str, Any] = fattrs["admin_data"]

                for i, (k, v) in enumerate(params.items()):
                    # get param module
                    param = self.g.G.nodes[k]
                    module = param["module"]

                    # add admin_data
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

    def handle_mod_stack(
            self,
            files:list[Any],
            root
    ):
        """
        receive files
        write to temp
        load in ModuleCreator -> create modules
        """
        print("handle_mod_stack")
        for f in files:
            # Pflichtfelder
            fname = f["name"]
            rel_path = f.get("path", fname)
            raw = base64.b64decode(f["admin_data"])
            target = root / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)

        # create modules form files -> save converted in G
        self.mcreator.main(
            temp_path=root
        )
        print("handle_mod_stack finished successfully")


    def create_db(self):
        """
        collect all nodes values and stack with axis def
        -> uniform for all envs
        """
        self.db = self.get_empty_field_structure()

        modules = self.g.get_nodes(
            filter_key="type",
            filter_value="MODULE",
        )

        for mid, m in modules:
            fields = self.g.get_neighbor_list(
                node=mid,
                target_type="FIELD",
            )

            if len(fields) > 0:
                #print(f"Fields for {mid} : {fields}")
                for fid, fattrs in fields.items():
                    self.db[m["module_index"]][fattrs["field_index"]] = [
                        fattrs["value"],
                        fattrs["axis_def"]
                    ]
            else:
                print(f"NO fields for {mid}: {fields}")


    def compile_pattern(self):
        try:
            compiler_struct = self.get_empty_field_structure()

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

                for fid, fattrs in fields.items():
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

                        for pid, pattrs in params.items():
                            pval = pattrs["value"]

                            # SELF PARAM
                            if pid in keys:
                                pindex = keys.index(pid)

                                axis_def.append(faxis_def[pindex])
                                method_param_collector.append(
                                        [
                                            j,
                                            pindex,
                                            nfield_index,
                                        ]

                                )

                            # ENV ATTR
                            elif pid in self.env:

                                pindex = list(self.env.keys()).index(pid)
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
                                # NEIGHBOR VALUE
                                for j, module in enumerate(modules):
                                    nmkeys: list[str] = module["keys"]
                                    if pid in nmkeys:
                                        pindex = nmkeys.index(pid)

                                        # loop neighbors of field
                                        fneighbors = self.g.get_neighbor_list(
                                            node=fid,
                                            target_type="has_finteractant",
                                        )

                                        # get neighbors from field
                                        # mark: nfield-index represents row
                                        # of GT
                                        for nfid, nfattrs in fneighbors.items():
                                            nfield_index = nfattrs["field_index"]
                                            nfaxis_def: list[str] = module["axis_def"]

                                            # add axis def
                                            axis_def.append(
                                                nfaxis_def[nfield_index]
                                            )

                                            # append method to
                                            method_param_collector.append(
                                                [
                                                    j,
                                                    pindex,
                                                    nfield_index,
                                                ]
                                            )

                        return_index_map = [
                            i,
                            keys.index(eqattrs["return_key"]),
                            field_index,
                        ]

                        # save equation interaction pattern
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
                    print(f"EQ interaction pattern set up", self.module_pattern_collector)
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
        meq = sorted(meq.values(), key=lambda x: x["method_index"], reverse=True)
        return meq

try:
    import ray
    from ray import get_actor
    @ray.remote
    class GuardWorker(Guard):
        def __init__(
            self,
            qfu,
            g,
            user_id
        ):
            Guard.__init__(
                self,
                qfu,
                g,
                user_id
            )
except Exception as e:
    print("Ray not accessible:", e)



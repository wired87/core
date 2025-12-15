import os
import ray
from ray import get_actor

from _ray_core.base.admin_base import RayAdminBase
from _ray_core.globacs.state_handler.main import StateHandler
from fb_core.real_time_database import FBRTDBMgr

from app_utils import DOMAIN, USER_ID, TESTING, ENVC
from gnn_master.node_refactored import Node
from guard import GuardWorker

from qf_utils.all_subs import ALL_SUBS
from qf_utils.field_utils import FieldUtils
from qf_utils.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils
from utils.utils import Utils


class Relay(
    Utils,
    FieldUtils,
    RayAdminBase,
):
    """
    relay implemtn agent to
    ORGANIZER talks to god and so on

    TODO create edge tables anfd upsert edata for each new connection
    paramG
    gnn wf
    dev & test new pw

    each created edge in bq,
    load edge state in spg

    design it like
    indem ich auf zeitsprünge überschpringen kann, kann cih mich überall in ruamund zeit bewegen

    Get data from change stream

    filter table type

    send to gpu worker
    env,
    ntype,
    edge_ids,
    host,
    Upsert initial ncfg
    Test run
    todo field store auf keinen fall in graph speichern inkl edges
    graph nur px und variablen
    """

    def __init__(
            self,
            world_cfg,
    ):
        super().__init__()

        self.utils = get_actor("UTILS_WORKER")
        self.sh = StateHandler()

        self.interactive_field_neighbors = None
        self.id = "relay"

        self.device = "gpu" if TESTING is False else "cpu"

        self.get_neighbors_endp = "/get-neighbors"
        self.get_table_entry_endp = "/get-table-entry"

        self.domain = "http://127.0.0.1:8000" if os.name == "nt" else f"https://{DOMAIN}"

        self.spanner_worker_url = f"{self.domain}/sp/create-rcs/"

        self.node=None
        self.px_pos_map = {}  # id:pos

        self.module_store=None
        self.firebase = FBRTDBMgr()
        self.deploy_modules = True # todo implement
        self.arsenal_struct = {}

        self.db_swat_set = False

        self.world_cfg=world_cfg
        self.sim_time_s = self.world_cfg["sim_time_s"]
        self.time = 0
        self.cluster_ncfg = None
        self.modules = {}

        self.g = GUtils(USER_ID, G=self.get_G())
        self.finished_fworkers = False
        self.finished_mworkers = False
        self.ready_map = {sub: False for sub in ALL_SUBS}

        print("RELAY INITIALIZED")


    def prepare(self):
        print("======== RELAY PREPARE ========")
        try:
            self.g = GUtils(
                user_id=USER_ID,
                G=self.get_G()
            )
            self.qfu = QFUtils(self.g)
            self.main()
        except Exception as e:
            print("Err prepare relay:", e)


    def main(self):
        """
        Iter for each ntype
        """
        print("======== RELAY MAIN ========")
        try:
            # todo relay ahndles cration

            # deploy GUARD
            self.deploy_guard()

            self.create_gpu_updator()

        except Exception as e:
            print("Err RELAY.main:", e)






    def sim_finisher(
            self,
            gid,
s    ):
        try:
            # todo before ups filter attrs
            # todo make data avaialble shets connect bq
            # l todo visualizie just when requested -> create programatic data shematic
            # l todo apply ml -> host vertex
            print(f"RELAY FINISHED: {gid}")

            print("RM GUARD . . .")
            ray.kill(gid)
            print("KILL CONFIRMED")

            self.shutdown_sys()
        except Exception as e:
            print("Err round_finisher: ", e)


    def apply_t_increase(self, nid_map):
        """
        NCFG Proccess
        Update sim time
        """
        print("start RELAY._initialize_and_check")
        if nid_map:
            return nid_map
        self.sh.await_alive(["UTILS_WORKER"])

        # Increment time if simulation is still running
        if self.time < self.world_cfg["sim_time_s"]:
            self.time += 1
            print("GLOBAC_STORE")
            # Print global store for debugging
        else:
            pass
        return nid_map


    def await_state(self):
        state = None
        while state is None:
            ray.nodes_ready.remote()


    def create_gpu_updator(self):
        print("start RELAY.create_updators")
        # UPDATOR NODE (GPU)
        self.g.add_node(
            {
                "nid":"UPDATOR",
                "type": "NODE",
                "ref": Node.options(
                    lifetime="detached",
                    name=f"UPDATOR",
                ).remote(
                    env=ENVC,
                    device=self.device,
                    max_index=len(ALL_SUBS),
                    amount_nodes=self.world_cfg["amount_nodes"],
                )
            }
        )
        self.sh.await_alive(["UPDATOR"])
        print("finished module_updaes")



    def get_px_and_subs(self, id_map):
        """
        Extract the parent pixel to each changed field
        """
        # px_id:list[nnid]
        px_changed_node_struct = {}
        missing_pos = []
        for nid in id_map:
            px_id = f"px{nid.split('__px')[1]}"
            if px_id not in self.px_pos_map:
                missing_pos.append(px_id)

            #
            if px_id not in px_changed_node_struct:
                px_changed_node_struct[px_id] = []
            px_changed_node_struct[px_id].append(nid)
        return px_changed_node_struct





"""
    def trigger_module_creation(
            self
    ) -> Callable:
        try:
            print("start RELAY._process_and_trigger_pathway")

            # await mworkers alive
            self.sh.await_alive(
                list(self.modules.keys())
            )

            # await fworkers alive
            self.sh.await_alive(ALL_SUBS)

            # fworkers finished init?
            all_fw_finished = self.all_nodes_ready(
                ALL_SUBS
            )


            # bring them to exec orderd format
            ordered_methods = self.get_execution_order(
                arsenal_methods
            )

            # extract pattern
            for ntype in ALL_SUBS:
                parent = self.parent_ntype(ntype)
                for item in self.couplings[parent].items():

            ntype_index_pattern_map = []
            for ntype in ALL_SUBS:
                print("module", ntype)
                try:
                    field_index_map = []
                    for nntype in self.coupling_partners[ntype]:
                        field_index_map.append(ALL_SUBS.index(nntype))

                    ntype_index_pattern_map.append(field_index_map)

                except Exception as e:
                    print(f"Err RELAY._process_and_trigger_pathway: {e}")
            print("end RELAY._process_and_trigger_pathway")

            get_actor("UPDATOR").main.remote(
            )

        except Exception as e:
            print("Err RELAY._process_and_trigger_pathway: ", e)



"""

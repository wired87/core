import os
import zipfile
from io import BytesIO
from tempfile import TemporaryDirectory

from qf_core_base.qf_utils.field_utils import FieldUtils
from qf_core_base.qf_utils.qf_utils import QFUtils

from qf_core_base.qf_utils.stimmulator.stimulator import Stimulator
from utils.file._csv import dict_2_csv_buffer

from utils.file.zip import _create_zip_archive


from utils.graph.local_graph_utils import GUtils
from utils.id_gen import generate_id
from utils.logger import LOGGER

import dotenv
dotenv.load_dotenv()
USER_ID = os.environ.get("USER_ID")
ENV_ID = os.environ.get("ENV_ID")
DEBUG = os.name == "nt"

class SimCore:


    """
    Handler fpr test, demo and prod sim runs.
    """

    def __init__(
            self,
            env_id=None,
            env_c=None,
            user_id=USER_ID,
            g_path=None, # local path || db path || None (demo)
            visualize=False,
            train_gnn=False,
            demo=True,
            sim_iters: int = None,
            g=None,
            mode: str = "demo",  # demo | test | prod
    ):
        self.bz_content = None

        self.stimulator = None
        self.env_id = env_id
        self.g_path = g_path
        self.env_c = env_c
        self.user_id = user_id
        self.testing = True
        self.demo = demo

        self.buffer = BytesIO()
        self.file_store = TemporaryDirectory()

        self.sim_iters = 10  # if demo is True else sim_iters
        self.zip_store = zipfile.ZipFile(self.buffer, mode="w")

        self.state = {
            "demo": os.name != "nt",
            "testing": True
        }

        self.field_utils = FieldUtils()

        self.run_id = generate_id()
        self.run = True
        self.user_id = user_id

        # Path handling -> temp content gets written to this dir
        self.save_base = f"qf_sim/_database/{self.user_id}/{self.run_id}"
        self.save_g_visual = os.path.join(self.save_base, "graph.html")

        # classes
        if g is None:
            self.g = GUtils(
                nx_only=True,
                g_from_path=None,
                user_id=user_id,
                enable_data_store=True
            )
        else:
            self.g = g

        self.train_gnn=train_gnn
        """if train_gnn is True:
            self.model_builder = GraphPredictor(
                self.g.datastore,
                save_path=os.path.join(
                    self.file_store.name,
                    "predictor.pt",
                )
            )"""

        self.qf_utils = QFUtils(
            self.g, user_id, env_id
        )
        self.visualize = True
        """if self.visualize is True:
            self.visualizer = Visualizer(
                self.file_store.name,
                qf_utils=self.qf_utils,
            )"""
        self.loop = 0
        print("SimCore initialized")

    def create(self, sim_cfg=None):
        # create
        print("Create Graph")

        #self.sim_runner.create()


        self._set_node_copy()
        self._set_edge_copy()



    def start_cluster(self):
        pass

    def _set_node_copy(self):
        print("_set_node_copy")
        self.frontend_nodes = self.g.get_node_pos()
        print("serializable_node_copy set:")
        #pprint.pp(serializable_node_copy)
        #self.serializable_node_copy[nid]["state"] = attrs.get("state")

    def _set_edge_copy(self):
        frontend_edges=self.g.get_edges_src_trgt_pos()
        self.frontend_edges = frontend_edges

    
    def create_archive(self):
        self.bz_content = _create_zip_archive(
            zip_name_without_tag=self.user_id,
            src_origin=self.file_store.name,
            rm_src_origin=True
        )

    def run_connet_test(self, dim=3, amount_nodes=2):
        if self.demo is True:
            self.create(
                {
                    "qf": {
                        "shape": "rect",
                        "dim": [amount_nodes for _ in range(dim)]
                    }
                }
            )
        else:
            self.qf_utils.fetch_db_build_G()

        # print("self.env_id", self.env_id)
        self.env = self.g.G.nodes[self.env_id]
        # pprint.pp(self.env)

        # time.sleep(10)
        # todo collect more sim data like len, elements, ...
        # todo improve auth
        if not self.user_id or not self.env_id:
            LOGGER.info(f"Connection attempt declined")
            return

        # init stimulator for autonomous stim apply
        self.stimulator = Stimulator(
            self.g,
            types_of_interest=["electron", "photon"],
            testing=self.testing,
            demo=self.demo
        )

        self.run_sim(self.g)

        # AI handler
        # self.all_qfn_pos = [{nid:attrs.get("pos")} for nid, attrs in self.g.G.nodes(data=True)]
        # self.msg_handler = MsgHandler(pos_list=self.all_qfn_pos)

    def run_sim(self, g: GUtils, env:dict=None):
        from qf_core_base.runner.qf_updator import QFUpdator
        if env is None:
            env=self.env

        self.updator = QFUpdator(
            g,
            env,
            self.user_id,
        )

        # todo @ testig start local docker and connect -> return endpoint when ready. rest keeps the same

        i = 0
        print("Start sim loop")
        while self.run is True:
            # print("run", i)
            i += 1
            self.updator.update_qfn()
            self._loop_event()

    ############# #
    ####### UTILS #
    ############# #

    def _loop_event(self):
        #self.stimulator.main()
        if self.loop >= self.sim_iters:
            self._finish()
            self.run = False
        else:
            self.loop += 1

    def _finish(self):
        #self._finisher()
        self.loop = 0
        self.run = False

    def _finisher(self):
        # todo DBManager
        """
        Create animation
        Train model
        Save G data local
        Save html data local
        """

        # print("start finisher")

        if self.g.enable_data_store is True:
            """# Create visuals (single field plots and G animation)
            if self.visualize is True:
                self.visualizer.main()"""

            # GNN Superpos predictor -> build -> train -> save to tmp
            """
            if self.train_gnn is True:
            self.model_builder.main()
            """

        dict_2_csv_buffer(
            data=self.updator.datastore,
            keys=self.g.id_map,
            save_dir=os.path.join(f"{self.file_store.name}", "datastore.csv")
        )

        self.create_archive()


if __name__ == "__main__":
    sc = SimCore()
    g = GUtils(
        nx_only=False,
        G=None,
        g_from_path=None,
        user_id=USER_ID,
    )

    ###
    sc.env = g.G.nodes[ENV_ID]
    print("QFUpdator ENV:", sc.env)
    sc.run_sim(g)


r"""
👉 Ions create electric fields by moving around.
👉 Electric fields guide ions.
👉 This feedback loop is what you simulate.
- the keys to intelligence lies in ion interaction
"""
import os
import time
from tempfile import TemporaryDirectory

from core.app_utils import USER_ID, ENV_ID
from data import QF_LEX
from qf_utils.all_subs import ALL_SUBS, G_FIELDS, H, FERMIONS
from qf_utils.mover import Mover
from qf_utils.qf_utils import QFUtils


class God:

    """
    HANDLES THE CREATION OF A CLUSTER
    Creator for bare empty sism World
    Start/Change ->
    Adapt Config ->
    Model Changes ->
    Results Upload Graph ->
    Render.
    """

    def __init__(
            self,
            g,
            qfu:QFUtils,

            env_id=ENV_ID,
            user_id=USER_ID,
            testing=False,
            world_type="bare",
            enable_data_manager=True,
            file_store=None,
    ):
        print("init God")
        self.bz_content = None
        self.user_id = user_id
        self.world_type=world_type

        self.enable_data_manager = enable_data_manager

        self.spread_items_type = [
            "PIXEL"
        ]

        self.qfu=qfu
        self.save_path = rf"C:\Users\wired\OneDrive\Desktop\qfs\qf_sim\world\graphs\world_{user_id}.json" if os.name == "nt" else f"qf_sim/world/graphs/world_{user_id}.json"
        self.g = g
        self.raw = True  # upload without linking anything
        self.filter_for = "EXPERIMENT_accession_"
        self.size = 0
        self.batch_size = 10
        self.testing = testing
        self.demo_g_save_path = r"qf_sim/_database/demo"
        self.file_store = file_store or TemporaryDirectory()

        """self.px_creator = PixelCreator(
            user_id=USER_ID,
            G=self.g.G,
        )"""

        self.env_id = env_id


        self.image_path = self.get_g_dest()
        self.mover = Mover(self.g)
        self.run_batch_gcp = True
        self.current_file = None
        print("Relay initialized")


    def create_lexicon(self):
        # Load G instance in G
        self.g.add_node(
            attrs=dict(
                nid="self",
                type="CLASS_INSTANCE",
                instance=self,
            )
        )

        for nid, meta_attrs in QF_LEX.items():
            print("create PARAM:", nid)
            self.g.add_node(
                attrs=dict(
                    nid=nid,
                    type="PARAM",
                    **{
                        k:v
                        for k, v in meta_attrs.items() if k not in ["nid", "type"]}
                )
            )



    def append_meta_nodes(self, nid_list, parent, meta_map):
        for nid in nid_list:
            node_header: dict = self.g.get_node_attrs_core(
                nid,
                ntype=nid.split("__")[0],
                parent=parent,
            )
            for key, value in node_header.items():
                meta_map[key].append(value)


    def finish(self):
        """
        Create media
        """
        # Just local save here (no dbn upsert and no plots)
        if os.name == "nt":
            path = r"C:\Users\bestb\Desktop\qfs\outputs\demo_G.json"
        else:
            path = "outputs/demo_g.json"

        # if no demo graph - save
        if not os.path.isfile(path):
            self.demo_g_save_path = r"outputs/demo_g.json"
            print("Save demo to", self.demo_g_save_path)
            self.g.save_graph(
                dest_file=self.demo_g_save_path  # if OS_NAME == "nt" else
            )

        # Create G html
        self.g.create_html()



    def get_g_dest(self):
        if os.name == "nt":
            return rf"C:\Users\wired\OneDrive\Desktop\qfs\qf_sim\physics\quantum_fields\nodes\qf\graphs\g{self.user_id}.html"
        return f"qf_sim/physics/quantum_fields/nodes/qf/graphs/g{self.user_id}.html"


    def connect_meta_nodes(self):
        print("Connect nodes")
        for nid, args in self.g.G.nodes(data=True):
            for nnid, nargs in self.g.G.nodes(data=True):
                # Just connect Meta/Parent objects here (QF- not QFN)
                src_layer = args.get('type')
                trgt_layer = nargs.get('type')
                if nid != nnid and (src_layer == "ENV" or src_layer == "QF") and (trgt_layer == "ENV" or trgt_layer == "QF"):
                    self.g.add_edge(
                        src=nid,
                        trgt=nnid,
                        attrs=dict(
                            src_layer=src_layer,
                            trgt_layer=trgt_layer,
                            rel="related",
                        )
                    )

        # remove the path specs from each node
        for nid, args in self.g.G.nodes(data=True):
            if args.get("type") not in ["USERS", "PARAMETER", "EQUATION"]:
                if "EC" in args.keys():
                    args.pop("EC")
                    self.g.G.nodes[nid].update(args)
        print("All Parent Nodes Connected")


    def load_attrs_in_sp(self):
        """
        CPU admin_data creation - create spanner
        """


    def load_attrs_in_G(self, world_cfg: dict,):
        """
        Creates params vertical inside the graph

        Genrate wentire world content
        todo ram will fuck in -> you must scale to spanner
        """
        self.world_cfg=world_cfg
        self.cluster_dim = world_cfg["cluster_dim"]
        start_time = time.monotonic()
        print("Scale Params vertical inside G:", )
        all_pixel_nodes = [
            (nid, args)
            for nid, args in self.g.G.nodes(data=True)
            if args.get("type") == "PIXEL"
        ]

        default_px_id = "px_0"
        #self.g.print_status_G("G BEFORE LOADUP")

        # todo save instant in vertical format
        for ntype in ALL_SUBS:
            try:
                self.g.add_node(
                    dict(
                        nid=ntype,
                        type="FIELD",
                        finished_run=False
                    )
                )

                #print("AMOUNT N:", self.px_creator.amount_nodes)
                for i in range(self.world_cfg["amount_nodes"]):
                    px_id = f"px_{i}"
                    args = self.g.get_node(
                        nid=px_id
                    )

                    pos:list[float, int] = args["pos"]

                    attrs: list[dict] = self.qfu.get_attrs_from_ntype(
                        ntype=ntype,
                        px_id=px_id,
                        pos=pos
                    )

                    for item in attrs:
                        #print("item", item)
                        item.update({"npm": npm})

                        self.g.add_node(
                            attrs=item
                        )

                        #print("Added node", item["nid"])
            except Exception as e:
                print("Err load_attrs_in_G", e)

        end_time = time.monotonic()
        duration = end_time - start_time
        #self.g.print_status_G("G AFTER LOADUP")
        print(f"\nRelay finished field attrs in {duration:.4f} seconds")


    def get_nid_map(self) -> dict:
        nid_map = {
            "FERMION":[],
            "GAUGE":[],
            "HIGGS":[],
        }

        for ntype in ALL_SUBS:
            # get module parent
            parent = None
            ntype=ntype.lower()
            if ntype in FERMIONS:
                parent = "FERMION"
            elif ntype in G_FIELDS:
                parent = "GAUGE"
            elif ntype in H:
                parent = "HIGGS"

            if parent is None:
                raise ValueError(f"Invalid nid_map parent for {ntype}")

            for i in range(self.world_cfg["amount_nodes"]):
                if "quark" in ntype.lower():
                    for item_index in range(3):
                        nid_map[parent].append(
                            f"{ntype}__px_{i}__{item_index}"
                        )
                elif "gluon" in ntype.lower():
                    if "gluon" == ntype.lower():
                        for item_index in range(8):
                            nid_map[parent].append(
                                f"{ntype}__px_{i}__{item_index}"
                            )
                else:
                    nid_map[parent].append(
                        f"{ntype}__px_{i}"
                    )
            print("nid_map created:", nid_map)
        return nid_map



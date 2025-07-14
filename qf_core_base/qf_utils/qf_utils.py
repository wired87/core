import os

from qf_core_base.fermion import FERM_PARAMS
from qf_core_base.g import GAUGE_FIELDS
from qf_core_base.qf_utils.all_subs import FERMIONS
from qf_core_base.qf_utils.field_utils import FieldUtils

from utils.graph.local_graph_utils import GUtils
from utils.file._csv import dict_2_csv

from gdb_manager.g_utils import DBManager

class QFUtils(FieldUtils):

    def __init__(self, g: GUtils, user_id=None, env_id=None, testing=False):
        super().__init__()
        self.g = g
        self.field_utils = FieldUtils()
        self.user_id = user_id
        self.env_id = env_id
        self.database = f"users/{self.user_id}/env/{self.env_id}/"
        self.testing = testing

        if self.testing is True:
            self.db_manager = DBManager(
                upload_to="fb",
                instance=os.environ.get("FIREBASE_RTDB"),  # set root of db
                database=self.database,  # spec user spec entry (like table)
                nx_only=False,
                G=None,
                g_from_path=None,
                user_id=self.user_id,
            )

    def extract_model_data(self):
        for nid, attrs in [
            (nid, attrs) for nid, attrs in self.g.datastore.nodes(data=True)
            if attrs.get("base_type").upper() in self.all_sub_fields]:
            # extract importantfields
            ntype = attrs.get("type")
            if ntype in FERMIONS:
                pass
                # todo

    def save_G_data_local(self, data, keys, path):
        dict_2_csv(data=data, keys=keys)

    def fetch_db_build_G(self):
        self.initial_frontend_data = {}

        initial_data = self.db_manager._fetch_g_data()
        if initial_data is None:
            self.state = {
                "msg": "unable_fetch_data",
                "src": self.db_manager.database,
            }
        # Build a G from init data and load in self.g
        self.g.build_G_from_data(initial_data, self.initial_frontend_data, self.env_id)

    def get_all_node_sub_fields(self, nid):
        # get intern qf parents
        phi_id, phi_attrs = self.g.get_single_neighbor_nx(
            node=nid,
            target_type="PHI"
        )
        psis = self.g.get_neighbor_list(
            node=nid,
            target_type=[k.upper() for k, v in FERM_PARAMS.items()]
        )
        gs = self.g.get_neighbor_list(
            node=nid,
            target_type=[k.upper() for k, v in GAUGE_FIELDS.items()]
        )
        # michael.kobel@tu-dresden.de
        return [
            [(phi_id, phi_attrs)],
            psis,
            gs
        ]

    def get_all_subs_list(self, check_key="type", datastore=False, just_attrs=False, sort_for_types=False, sort_index_key="entry_index"):
        if datastore is True:
            all_subs:list[dict or tuple] = self.get_all_field_nodes(
                self.g.datastore, check_key
            )
        else:
            all_subs:list[dict or tuple] = self.get_all_field_nodes(
                self.g.G, check_key
            )

        sorted_node_types={}

        if sort_for_types == True :
           #print("Sort nodes for types")
            for nid, attrs in all_subs:
                # save for type
                ntype = attrs.get("type")
                if ntype not in sorted_node_types:
                    sorted_node_types[ntype] = []

                # Deserialize Node
                converted_dict = self.field_utils.restore_selfdict(attrs)

                sorted_node_types[ntype].append(converted_dict)

            # Sort nodes newest -> oldest
            if sort_index_key is not None:
               #print("Sort nodes for index")
                for node_type, rows in sorted_node_types.items():
                    new_rows = sorted(rows, key=lambda d: d[sort_index_key])
                    sorted_node_types[node_type] = new_rows

           #print("Return nodes")
            return sorted_node_types

        else:
            if just_attrs is True:
                return [attrs for nid, attrs in all_subs]
            else:
                return all_subs

    def get_all_field_nodes(self, G, check_key):
        return [(nid, attrs) for nid, attrs in G.nodes(data=True) if
                attrs.get(check_key).upper() in self.all_sub_fields]


    def create_connection(
            self,
            node_data: list,
            coupling_strength,
            env_id,
            con_type: str,  # "TRIPPLE" | "QUAD"
            nid
    ):
        # Does the tripple already exists?
        tripples = self.g.get_neighbor_list(nid, con_type)
        tripple_exists = True
        for tid, tattrs in tripples:
            for n in node_data:
                if n["id"] not in tid:
                    tripple_exists = False
                    break
        # No! -> create
        if tripple_exists is False:
            node_id = "_".join(n["id"] for n in node_data)
            ntype = con_type
            self.g.add_node(
                attrs={
                    "id": node_id,
                    "coupling_strength": coupling_strength,
                    "type": ntype,
                    "ids": node_data,
                }
            )

            # Connect tripple to ENV
            self.g.add_edge(
                src=env_id,
                trt=node_id,
                attrs=dict(
                    rel=f"has_{con_type.lower()}",
                    src_layer="ENV",
                    trgt_layer=ntype
                )
            )

            # Connect nodes to tripple
            for item in node_data:
                self.g.add_edge(
                    src=node_id,
                    trt=item["id"],
                    attrs=dict(
                        rel=ntype.lower(),
                        src_layer=ntype,
                        trgt_layer=item["type"]
                    )
                )
           #print(f"{con_type} created!")
        else:
           print("Tripple already exists")






from app_utils import ENV_ID
from fb_core.real_time_database import FirebaseRTDBManager
from qf_core_base.fermion import FERM_PARAMS
from qf_core_base.g import GAUGE_FIELDS
from qf_core_base.qf_utils.all_subs import FERMIONS, G_FIELDS, ALL_SUBS
from qf_core_base.qf_utils.field_utils import FieldUtils

from utils.graph.local_graph_utils import GUtils
from utils.file._csv import dict_2_csv


class QFUtils(FieldUtils):

    def __init__(self, g: GUtils=None, user_id=None, env_id=None, testing=False):
        super().__init__()
        self.g = g
        self.field_utils = FieldUtils()
        self.user_id = user_id
        self.env_id = env_id
        self.metadata_path = "metadata"
        self.testing = testing

        if self.testing is True:
            self.db_manager=FirebaseRTDBManager()


    def change_state(self):
        """Changes state of ALL metadata entries"""
        upsert_data = {}
        data = self.db_manager.get_data(path=self.metadata_path)
        #pprint.pp(data)
        for mid, data in data["metadata"].items():
            current_state = data["status"]["state"]
            if current_state == "active":
                new_state = "inactive"
            else:
                new_state = "active"
            upsert_data[f"{mid}/status/state/"] = new_state

        self.db_manager.update_data(
            path=self.metadata_path,
            data=upsert_data
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

        initial_data = self.db_manager._fetch_g_data(
            db_root=f"users/{self.user_id}/env/{ENV_ID}/"
        )

        # Build a G from init data and load in self.g
        self.g.build_G_from_data(initial_data, self.env_id)

    def get_all_node_sub_fields(self, nid, edges=False, classify_in_ntype=False):
        # get intern qf parents
        phi = self.g.get_neighbor_list(
            node=nid,
            target_type="PHI",
        )
        psis = self.g.get_neighbor_list(
            node=nid,
            target_type=[k.upper() for k, v in FERM_PARAMS.items()],
        )
        gs = self.g.get_neighbor_list(
            node=nid,
            target_type=[k.upper() for k, v in GAUGE_FIELDS.items()],
        )
        # michael.kobel@tu-dresden.de

        all_subs = {
            "PHI": phi,
            "FERMION": psis,
            "GAUGE": gs
        }

        if edges is True:
            all_edges = self.edges_for_subs(
                nid,
                all_subs,
            )
            all_subs["edges"] = all_edges

        if classify_in_ntype is True:
            # for vertex and updator classes the needed format is
            # parent:ntype:nid:attrs
            for field_type in all_subs.keys():
                field_type_struct = {}
                for nid, attrs in all_subs[field_type].items():
                    ntype = attrs.get("type")
                    if ntype not in field_type_struct:
                        field_type_struct[ntype] = {}
                    field_type_struct[ntype][nid] = attrs
                all_subs[field_type] = field_type_struct



        return all_subs

    def get_ids_from_struct(self, all_subs):
        node_ids = []
        edge_ids = []
        for field_type, ntype in all_subs.items():
            if field_type.lower() == "edges":
                edge_ids.extend(
                    list(ntype.keys())
                )
                continue
            for ntype, nnids in ntype.items():
                node_ids.extend(
                    list(nnids.keys())
                )
        return node_ids, edge_ids




    def edges_for_subs(self, nid, all_subs: list[list[tuple]] or dict) -> dict:
        """
        Uses the return value of get_all_node_sub_fields to get
        all edges of that connections and save it in
        dict: id: eattrs  -fromat
        print("field_type", field_type)
        print("ntype", ntype)
        print(f"<<<<<NTYPE: {nntype}")
        print(f"<<<<<node_attrs: {node_attrs}")
        """
        #print(f"Get edges for {nid}")
        all_edges = {}
        for field_type, ntype in all_subs.items():
            node_id_list = list(ntype.keys())
            #print("node_id_list", node_id_list)
            if len(node_id_list):
                for nnid in node_id_list:
                    edge_attrs = self.g.G.edges[nid, nnid]
                    eid = edge_attrs.get("id")
                    all_edges[eid] = edge_attrs

        return all_edges



    def get_all_subs_list(self, check_key="type", datastore=False, just_attrs=False, just_id=False, sort_for_types=False, sort_index_key="entry_index", return_dict=False):
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
                if return_dict is False:
                    sorted_node_types[ntype].append(converted_dict)
                else:
                    sorted_node_types[ntype][nid] = converted_dict

            # Sort nodes newest - > oldest
            if sort_index_key is not None:
               #print("Sort nodes for index")
                for node_type, rows in sorted_node_types.items():
                    new_rows = sorted(rows, key=lambda d: d[sort_index_key])
                    sorted_node_types[node_type] = new_rows

            #print("Return nodes")
            return sorted_node_types

        else:
            if just_attrs is True:
                return [attrs for _, attrs in all_subs]
            elif just_id is True:
                return [nid for nid, _ in all_subs]
            else:
                return all_subs

    def get_all_field_nodes(self, G, check_key):
        return [(nid, attrs) for nid, attrs in G.nodes(data=True) if
                attrs.get(check_key).upper() in self.all_sub_fields]


    def list_subs_ids(self):
        all_subs = [nid for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type").upper() in [*ALL_SUBS, "PIXEL", "PX"]]
        return all_subs



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


    def get_npm_values(self, npm:dict, ntype, field_key=None):
        """
        Extract all values from neighbo_pm-struct
        """
        npm_val_struct= {}
        for dir, (np, nm) in npm.items():
            npm_val_struct[dir] = {}

            if field_key is None:
                field_key = self.get_field_key(ntype)

            nm_fv = self.g.get_single_neighbor_nx(np, ntype)[1][field_key]
            np_fv = self.g.get_single_neighbor_nx(nm, ntype)[1][field_key]

            # use tuple because if plus minus = None, dict just gets singel mnus entry
            npm_val_struct[dir] = [
                (np, np_fv),
                (nm, nm_fv)
            ]
        #print("Neighbor PM vals extracted")
        return npm_val_struct


    def get_field_key(self, ntype):
        if ntype.lower() in FERMIONS:
            return "psi"
        elif ntype.lower() in G_FIELDS:
            return self._field_value(type=ntype)
        elif ntype.lower() == "phi":
            return "h"




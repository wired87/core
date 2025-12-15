import os

import ray
from tempfile import TemporaryDirectory


from _ray_core.base.base import BaseActor
from app_utils import USER_ID
from qf_utils.qf_utils import QFUtils
from utils.file._csv import dict_2_csv_buffer, collect_keys
from utils.graph.local_graph_utils import GUtils


@ray.remote
class DataStoreWorker(BaseActor):

    """
    Data Requas and base processign for Dataprocesor(remote)
    """
    def __init__(self):
        BaseActor.__init__(self)
        self.datastore:dict[str, list[dict]] = {}
        self.filestore = TemporaryDirectory()


    def store_data(self, data_ref):
        data:list[dict] =  ray.get(data_ref)
        # format id:tid:data
        for attrs in data:
            nid = attrs["nid"]
            if nid not in self.datastore:
                self.datastore[nid] = []
            self.datastore[nid].append(attrs)

    def get_data_files(self):
        for nid, tid_struct in self.datastore.items():
            data_struct = [row for row in tid_struct.values()]
            dict_2_csv_buffer(
                keys=collect_keys(data_struct),
                data=data_struct,
                save_dir=os.path.join(self.filestore.name, nid),
            )
        return self.filestore.name

class StateDataWorker(BaseActor, QFUtils):

    """
    Spanner replacement
    filters the right neighbors for incoming ids
    """

    def __init__(self, G):
        self.g = GUtils(user_id=USER_ID, G=G)

        BaseActor.__init__(self)
        QFUtils.__init__(self, g=self.g)

    def initialize_data_structures(self):
        # Set up the main data containers for classified nodes and edges

        # Keep track of all created/processed node and neighbor IDs
        self.id_map = set()
        self.total_neighbor_id_map = set()
        self.neighbor_id_struct = {}

    def retrieve_new_processable_nodes(self, nid_list):
        """
        Receive neighbors for changed nodes
        Return classified nodes, sepparated into
        parent px ids ->
        neigbor px ids ->
        interactants ->
        """
        # todo instant classify into modules ->
        #  convert to soa
        # DATA STRUCTURES
        self.initialize_data_structures()

        if nid_list:
            # px: list[neighbors]
            px_struct:dict[str,list[str]] = self.classify_nid_list_to_px(
                nid_list
            )
            print("px_struct", px_struct)
            for px_id, changed_nids in px_struct.items():
                # EXTRACT PX NEIGHBORS
                #print("retrieve_new_processable_nodes px_id", px_id)
                px_neighbors:dict[str, dict] = self.g.get_neighbor_list_rel(
                    trgt_rel="px_neighbor",
                    node=px_id,
                )
                if len(px_neighbors):
                    for npx in px_neighbors:
                        # loop all ntypes
                        npx_id = npx[0]
                        print("npx_id", npx_id)
                        for nid in changed_nids:
                            #print("nid", nid)
                            ntype = nid.split("__")[0]

                            # retun its nids
                            self.add_node(
                                ntype.upper(),
                                npx_id,
                                nid,
                            )

                            # nid(for all items): list[]
                            # to map interactant edges
                            nid_is_struct: dict = self.get_qf_nid(
                                ntype.upper(),
                                npx_id
                            )

                            self.neighbor_id_struct.update(
                                nid_is_struct
                            )

                            # get interactive fields from nid
                            neighbors: list[str] = self.get_interactive_neighbors(ntype)

                            self.total_neighbor_id_map.update(list(nid))

            # das web muss von allen gehostet und kontrolliert werden
            # Wrap with ray Opbect ref
            #arsenal_ref = ray.put(self.arsenal_struct)
            neighbor_id_ref = ray.put(self.neighbor_id_struct)
            total_neighbor_id_map_ref = ray.put(self.total_neighbor_id_map)

            #edge_ref = ray.put(self.edge_struct)
            return neighbor_id_ref, total_neighbor_id_map_ref
        else:
            print("No nid_list provided")
        return None, None, None



    def add_node(self,ntype,px_id,nid):
        attr_struct = self.get_attrs_from_ntype(
            ntype,
            px_id,
            nid,
        )
        for attrs in attr_struct:
            self.g.add_node(
                attrs=attrs
            )

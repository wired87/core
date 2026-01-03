
import logging
import networkx as nx
from datetime import datetime
from typing import List, Dict, Tuple

from qf_utils.qf_utils import QFUtils
from core.module_manager.mcreator import ModuleCreator
from core.module_manager.ws_modules_manager.modules_lib import module_manager, generate_numeric_id
from core.fields_manager.fields_lib import fields_manager
from core.user_manager.user import UserManager


class SMManager:
    def __init__(self):
        self.module_db_manager = module_manager
        self.field_manager = fields_manager
        self.user_manager = UserManager()

    def main(self, user_id: str = "public"):
        """
        Upsert standard nodes and edges from QFUtils to BigQuery tables.
        """
        print("PROCESSING STANDARD MANAGER WORKFLOW")
        """
        if self._check_standard_stack(user_id):
            print("Standard stack already exists.")
            return
        """
        
        # Create module stack
        qf = self._initialize_graph()
        
        # Upsert Nodes and Edges
        self._upsert_graph_content(qf, user_id)

        self.user_manager.set_standard_stack(user_id)
        print("FINISHED SM WORKFLOW")

    def _check_standard_stack(self, user_id: str) -> bool:
        """Check if standard stack exists for the user."""
        return self.user_manager.get_standard_stack(user_id)


    def _initialize_graph(self) -> QFUtils:
        """Initialize QFUtils and load standard modules."""
        # Create separate graph for SM loading
        qf = QFUtils(G=nx.Graph())
        qf.build_interacion_G()
        
        module_creator = ModuleCreator(
            G=qf.g.G,
            qfu=qf
        )
        module_creator.load_sm()
        
        return qf

    def _upsert_graph_content(self, qf: QFUtils, user_id: str):
        """Upsert nodes and edges from the graph to BigQuery."""
        # 1. Upsert Nodes
        logging.info("Upserting standard nodes...")
        modules, fields = self._extract_nodes(qf, user_id)
        
        if modules:
            self.module_db_manager.set_module(modules, user_id)
        if fields:
            self.field_manager.set_field(fields, user_id)

        # 2. Upsert Edges
        logging.info("Upserting standard edges...")
        mfs, ffs = self._extract_edges(qf, user_id)
        
        if mfs:
            self.field_manager.link_module_field(mfs)
        if ffs:
            self.field_manager.link_field_field(ffs)

    def _extract_nodes(
            self,
            qf: QFUtils,
            user_id: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Extract module and field nodes from the graph."""
        modules = []
        fields = []

        for nid, attrs in qf.g.G.nodes(data=True):
            ntype = attrs.get("type")

            if ntype == "MODULE":
                # Get PARAM neighbors
                # Utilizing the helper from qf.g which is expected to be GUtils or similar
                params = qf.g.get_neighbor_list(
                    target_type="PARAM",
                    node=nid,
                    just_ids=True,
                )

                # Upsert Module
                module_data = {
                    "id": attrs.get("nid", nid),
                    "file_type": None,
                    "binary_data": None,
                    "code": attrs.get("code"),
                    "user_id": user_id,
                    "created_at": datetime.utcnow().isoformat(),
                    "params": params,
                    "status": "active",
                }
                modules.append(module_data)

            elif ntype == "FIELD":
                # Upsert Field
                keys = attrs.get("keys") or attrs.get("field_keys")
                values = attrs.get("values") or attrs.get("value")
                axis_def = attrs.get("axis_def")

                if keys is None or values is None:
                    print(f"Skipping malformed FIELD node {nid}. Keys/Values missing.")
                    continue

                field_data = {
                    "id": nid,
                    "keys": keys,
                    "values": values,
                    "axis_def": axis_def,
                    "user_id": user_id,
                    "status": "active",
                }
                fields.append(field_data)
        return modules, fields

    def _extract_edges(self, qf: QFUtils, user_id: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract edges from the graph."""
        mfs = []
        ffs = []

        for u, v, attrs in qf.g.G.edges(data=True):
            src_layer = attrs.get("src_layer")
            trgt_layer = attrs.get("trgt_layer")

            if src_layer == "MODULE" and trgt_layer == "FIELD":
                # Module -> Field
                data = {
                    "id": generate_numeric_id(),
                    "module_id": u,
                    "field_id": v,
                    "user_id": user_id
                }
                mfs.append(data)

            elif src_layer == "FIELD" and trgt_layer == "FIELD":
                # Field -> Field
                data = {
                    "id": generate_numeric_id(),
                    "field_id": u,
                    "interactant_field_id": v,
                    "user_id": user_id
                }
                ffs.append(data)
        
        return mfs, ffs

# Instantiate
sm_manager = SMManager()

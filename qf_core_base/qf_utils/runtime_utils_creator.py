from typing import List

import ray

from qf_core_base.qf_utils.all_subs import ALL_SUBS, G_FIELDS
from qf_core_base.qf_utils.field_utils import FieldUtils
from _ray_core.ray_validator import RayValidator

class RuntimeUtilsCreator(
    RayValidator,
    FieldUtils
):

    def __init__(self, g, database, host={}):
        self.host = host
        RayValidator.__init__(
            self,
            g_utils=g,
            host=self.host
        )
        FieldUtils.__init__(self)
        self.updator = None
        self.g = g
        self.database = database
        self.sub_content = {}
        self.env = None  # -> set before creation of workers

    def main(self):
        all_pixel_nodes = self.call(
            method_name="get_nodes",
            filter_key="type",
            filter_value="PIXEL"
        )

        pixel_utils = self.create_px_runtime_utils(all_pixel_nodes, env=ray.get(self.host["UTILS_WORKER"].call.remote(method_name="get_env")))

        # Create runtime utisl for all subs and catrgorize them in ntypes
        # ( e.g. 100 pixels: we get 100 electrons in dict A, ...
        self.create_field_worker_utils(pixel_utils)

    def create_field_worker_utils(self, pixel_utils):
        """
        Create utils for each single field
        """

        for pxu in pixel_utils:
            self.create_utils_stack_all_subs(
                pxu,
            )

    def create_px_runtime_utils(self, all_pixels, env):
        self.env=env
        pixel_utils = [
            self.create_px_utils(
                pixel_id,
                pxattrs=attrs
            )
            for i, (pixel_id, attrs) in enumerate(all_pixels)
        ]
        return pixel_utils


    def create_utils_stack_all_subs(
            self,
            pxu,
    ) -> List[dict]:
        _nid = None
        try:
            pixel_id = pxu["pixel_id"]
            px_subs:dict = ray.get(
                self.host["UTILS_WORKER"].call.remote(
                    method_name="get_neighbor_list",
                    node=pixel_id,
                    target_type=ALL_SUBS,
                )
            )

            for nid, attrs in px_subs.items():
                _nid = nid

                ntype = attrs.get("type")

                utils_stack = {
                    "node_utils": self.create_node_rtu(
                        nid,
                        ntype,
                        attrs,
                        pxu["neighbors_pm"],
                        pixel_id,
                    ),
                    "pixel_utils": pxu,
                }
                print(f"{nid} utils_stack created: {list(utils_stack.keys())}")

                self.sub_content[nid] = utils_stack

            print("NKEYS", self.sub_content.keys())

        except Exception as e:
            print(f"<{_nid}> Error create Fieldworkers: {e}")

    def create_node_rtu(
            self,
            nid,
            host,
            attrs,
            npm,
            pixel_id,
    ):
        #neighbor_types = [*ALL_SUBS, "ENV", "PIXEL"]

        ntype = attrs.get("type")

        try:
            #print("self.host", self.host)

            neighbor_pm_val_same_type = self.call(
                method_name="get_npm_values",
                npm=npm,
                ntype=ntype
            )

            #print("neighbor_pm_val_same_type")

            neighbor_pm_val_fmunu = None
            if ntype.lower() in G_FIELDS:
                neighbor_pm_val_fmunu = self.call(
                    method_name="get_npm_values",
                    npm=npm,
                    ntype=ntype,
                    field_key="F_mu_nu"
                )

            #print("neighbor_pm_val_fmunu")

            # get list of all neighbors classified in types
            # and edges
            all_subs: dict = self.call(
                method_name="get_all_node_sub_fields",
                nid=nid,
                edges=True,
                classify_in_ntype=True
            )

            # Extend all_subs with down quark types (just for up type quarks ->
            # for later doublet combination
            self.short_lower_type = ntype.lower().split("_")[0]

            # up, charm, top
            if self.short_lower_type in self.ckm:
                # Extend existing subs struct
                all_subs = self._add_quark_partners(
                    pixel_id,
                    all_subs
                )

            node_ids = self.call(
                method_name="get_neighbor_list",
                node=nid,
                target_type=[*ALL_SUBS, "PIXEL"],
                just_ids=True
            )

            edge_ids = self.call(
                method_name="get_edge_ids",
                src=nid,
                neighbor_ids=node_ids
            )

            #print("get_ids_from_struct")
            self_item_up_path, self_h_entry_item_up_path = self.get_item_upsert_paths(
                nid, ntype
            )
            #print("get_item_upsert_paths")

            runtime_utils = dict(
                id=nid,
                attrs={
                    "id": nid,
                    **{k: v for k, v in attrs.items() if k != "id"}
                },
                env=self.env,
                host=host.copy(),
                neighbor_pm_val_same_type=neighbor_pm_val_same_type,
                all_subs=all_subs,
                neighbor_pm_val_fmunu=neighbor_pm_val_fmunu,
                self_item_up_path=self_item_up_path,
                self_h_entry_item_up_path=self_h_entry_item_up_path,
                neighbor_node_ids=node_ids,
                edge_ids=edge_ids,
                parent_pixel_id=pixel_id,
            )
            #print("n_rtu created")

        except Exception as e:
            print(f"Error creating node runtime utils: {e}")
            runtime_utils = {}

        return runtime_utils

    def _add_quark_partners(self, pixel_id, all_subs):
        """
        Extract all down quarks for doublet sum/combinations
        """
        # extract lower pre type for quarks (up, down, etc.)
        for charm_type, ckm_struct in self.ckm[self.short_lower_type].items():
            quark_type_full = f"{charm_type}_quark".upper()
            nnid, nattrs = self.call(
                method_name="get_single_neighbor_nx",
                node=pixel_id,
                target_type=quark_type_full
            )

            #print(f"_add_quark_partners nnid, nattrs: {nnid, nattrs}")
            if quark_type_full not in all_subs["FERMION"]:
                all_subs["FERMION"][quark_type_full] = {}
            all_subs["FERMION"][quark_type_full][nnid] = nattrs
        return all_subs

    def get_paths_to_listen(self, all_subs: dict, edge_ids) -> list:
        global_db_listener_path = f"{self.database}/global_states/"

        db_paths = [
            global_db_listener_path,
        ]
        for field_type, ntype in all_subs.items():
            # print("field_type", field_type)
            # print("ntype", ntype)
            if field_type.lower() != "edge":
                for ntype, nnids in ntype.items():
                    db_paths.extend(
                        f"{self.database}/{ntype}/{nnid}"
                        for nnid in list(nnids.keys())
                    )

        db_paths.extend(
            f"{self.database}/edges/{eid}"
            for eid in edge_ids
        )

        print("All db listener paths extracted")
        return db_paths

    def get_item_upsert_paths(self, nid, ntype):
        self_h_entry_item_up_path = f"{self.database}/datastore/{nid}"
        self_item_up_path = f"{self.database}/{ntype}/{nid}/"
        return self_item_up_path, self_h_entry_item_up_path

    def build_G_local_or_fetch(self, db_manager, session_id):
        """
        build G from data set self ref and creates
        all sub nodes of a specific
        """
        # Build G and load in self.g todo: after each sim, convert the sub-ield graph back to this
        print("BUild Graph")
        demo_G_save_path = self.call(method_name="get_demo_G_save_path")

        initial_data = db_manager._fetch_g_data()

        # Build a G from init data and load in self.g
        self.call(
            method_name="build_G_from_data",
            initial_data=initial_data,
            env_id=session_id,
            save_demo=True,
        )
        print("Graph successfully build")


    def create_px_utils(self, pixel_id, pxattrs, env, host):
        if self.env is None:
            self.env = env
        print("create_px_utils")
        try:
            pixel_neighbors = ray.get(self.host["UTILS_WORKER"].call.remote(
                method_name="get_neighbor",
                nid=pixel_id,
                trgt_type="PIXEL",
                single=False
            ))
            #print("pixel_neighbors", pixel_neighbors)

            px_subs: dict = ray.get(self.host["UTILS_WORKER"].call.remote(
                method_name="get_neighbor_list",
                node=pixel_id,
                target_type=ALL_SUBS,
            ))

            #print("px_subs", len(list(px_subs.keys())))

            px_subs_ids = list(px_subs.keys())
            #print("px_subs_ids", px_subs_ids)

            # reset
            px_subs = None

            npm = ray.get(self.host["UTILS_WORKER"].get_npm.remote(
                node_id=pixel_id,
                self_attrs=pxattrs,
                all_pixel_nodes=pixel_neighbors,
                env=self.env
            ))

            return {
                "neighbors_pm": npm,
                # "pixel_id": pixel_id,
                "attrs": pxattrs,
                "node_ids": px_subs_ids,
                "host": host.copy(),
                "env": self.env,
            }
        except Exception as e:
            print(f"Error create pixel utils: {e}")
            return {}

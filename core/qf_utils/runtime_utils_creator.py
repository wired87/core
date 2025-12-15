"""
Adapt the entire logic from use of nx.Graph to use
of GCP Spanner Graph:
- replace entire graph query logic with spanner quer, available through local dict_store
- just fetch values vertical (col) based -> minimize overhead - get all times just values != 0
- inherit from the provided spanenr graph classes
"""

import inspect
from typing import List, Callable

import ray
from ray import get_actor

from qf_utils.all_subs import ALL_SUBS, FERMIONS
from _ray_core.utils.ray_validator import RayValidator

from qf_utils.qf_utils import QFUtils


class RuntimeUtilsCreator(
    RayValidator,
    QFUtils,
):

    def __init__(self, g):
        super().__init__(g_utils=g)
        QFUtils.__init__(self, g)
        self.updator = None
        self.g = g
        self.sub_content = {sub: {} for sub in ALL_SUBS}
        #self.g.print_status_G("RuntimeUtilsCreator")

    def all_subs(self, attrs):
        nid=attrs["nid"]
        type=attrs["type"]
        pixel_id=attrs["px"]

        print("set all_subs")
        all_subs = self.get_all_node_sub_fields(
            nid=nid,
            as_dict=True,
            edges=True,
        )

        self.short_lower_type = type.lower().split("_")[0]

        if self.short_lower_type in self.ckm:
            all_subs = self._add_quark_partners(pixel_id, all_subs, attrs)
        return all_subs

    def get_neighbor_node_ids(self, all_subs, attrs):
        node_ids, _ = self.get_ids_from_struct(
            all_subs=all_subs,
        )
        return node_ids


    def neighbor_h(self, attrs):
        #attrs = FERMION
        higgs_id = f"HIGGS__{attrs['px']}"
        return self.g.get_node(higgs_id)["h"]


    def g_neighbors_gg(self, attrs):
        try:
            filtered = [[],[]]

            neighbors = self.all_px_neighbors(
                attrs
            )

            valid_neighbors:list[str] = self.gauge_to_gauge_couplings[attrs["type"].lower()]

            g_neighbors = self.get_nids_from_pxid(
                npxs=neighbors,
                ntypes=valid_neighbors
            )

            print("gneighbors extracte:", len(g_neighbors))

            for nnid in g_neighbors:
                nattrs = self.g.get_node(nid=nnid)

                filtered[0].append(
                    nattrs["field_value"])

                filtered[1].append(
                    nattrs["g"])

            return filtered
        except Exception as e:
            print("Err gg_neighbors:", e)


    def neighbor_f(self, attrs):
        filtered = [[],[]]
        try:
            neighbors = self.all_px_neighbors(
                attrs
            )

            valid_neighbors: list[str] = self.gauge_to_fermion_couplings[attrs["type"].lower()]

            neighbors = self.get_nids_from_pxid(
                npxs=neighbors,
                ntypes=valid_neighbors
            )

            for n in neighbors:
                nattrs = self.g.get_node(nid=n)

                filtered[0].append(nattrs["psi"])
                filtered[1].append(nattrs["psi_bar"])

            return filtered
        except Exception as e:
            print("Err neighbor_f:", e)


    def g_neighbors(self, attrs):

        try:
            ntype = attrs.get("type").lower()
            # print("g_neighbors", ntype)
            # todo save new data structures
            filtered = [[], [], []]
            npxs:list[str] = self.all_px_neighbors(attrs)
            ntypes = self.fermion_to_gauge_couplings[ntype]

            g_neighbors = self.get_nids_from_pxid(
                npxs=npxs,
                ntypes=[n.upper() for n in ntypes]
            )

            for nnid in g_neighbors:
                args = self.g.get_node(nid=nnid)
                nntype = args["type"]
                filtered[0].append(
                    args["field_value"]
                )
                filtered[1].append(args["g"])

                filtered[2].append(
                    self._get_gauge_generator(
                        nntype,
                        gluon_index=args["gluon_index"],
                        quark_index=attrs["quark_index"],
                    )
                )
            #print("g_neighbors item:", filtered)
            return filtered
        except Exception as e:
            print("Err g_neighbors:", e)

    # todo coding sys based on components - not single code lines
    def h_neighbor(self, attrs):
        nid=attrs["nid"]
        return self.g.get_node(nid=f"PHI__px{nid.split('_px')[-1]}")

    def f_neighbors(self, attrs):
        nid=attrs["nid"]
        return self.g.get_neighbor_list(node=nid, target_type=[g.lower() for g in FERMIONS])

    def neighbors(self, attrs):
        nid=attrs["nid"]

        neighbor_types = [*ALL_SUBS, "ENV", "PIXEL"]
        neighbors: tuple = self.g.get_neighbor_list(node=nid, target_type=neighbor_types)
        return {n: a for n, a in neighbors}

    def attrs(self, attrs, nid=None):
        return {
            "id": nid or attrs.get("id"),
            **{k: v for k, v in attrs.items() if k != "id"}
        }


    def create_field_worker_utils(self, pixel_utils, attrs):
        for pxu in pixel_utils:
            self.create_utils_stack_all_subs(pxu, attrs)

    def get_step_eq_val(self, attrs, method_key):
        print("get_step_eq_val", method_key)
        try:
            method=getattr(self, method_key)
            if isinstance(method, Callable):
                #print("method", type(method))
                # LOOP METHOD PARAM REQs
                if len(self.get_method_params(method)):
                    result = method(
                        attrs
                    )
                else:
                    result = method()
            else:
                print(f"{method_key}:", type(method))
                result = method
            return result

        except Exception as e:
            print("Err get_step_eq_val", e, "method_key", method_key)


    def get_method_params(
            self,
            methoden_objekt,
    ) -> list:
        #todo each step gets new edges
        """

        methoden_objekt = getattr(
            updator,
            methoden_name
        )
        """

        # 3. Die Signatur der Methode erhalten
        signatur = inspect.signature(methoden_objekt)

        # 4. Nur die Parameter-Namen (ohne 'self') extrahieren
        parameter_names = [
            param.name for param in signatur.parameters.values() if param.name != 'self'
        ]
        #print("param names extracted:", parameter_names)
        return parameter_names


    def create_utils_stack_all_subs(self, pxu, attrs) -> List[dict]:
        try:
            pixel_id = pxu["pixel_id"]
            all_px_subs = ray.get(get_actor(
                "UTILS_WORKER"
            ).get_nodes.remote(
                filter_key="type",
                filter_value=ALL_SUBS,

            ))
            for nid, nattrs in all_px_subs.items():
                ntype = nattrs.get("type")
                self.sub_content[nid] = {
                    "node_utils": self.create_node_rtu(
                        nid, ntype, nattrs, pxu["npm"], pixel_id, attrs
                    ),
                    "pixel_utils": pxu,
                }
        except Exception as e:
            print(f"Error create Fieldworkers: {e}")

    def _add_quark_partners(self, pixel_id, all_subs, attrs):
        for charm_type, ckm_struct in self.ckm[self.short_lower_type].items():
            quark_type_full = f"{charm_type}_quark".upper()
            nnid, nattrs = self.call(
                method_name="get_single_neighbor_nx",
                node=pixel_id,
                target_type=quark_type_full,
            )
            if quark_type_full not in all_subs["FERMION"]:
                all_subs["FERMION"][quark_type_full] = {}
            all_subs["FERMION"][quark_type_full][nnid] = nattrs
        return all_subs


    def build_G_local_or_fetch(self, db_manager, session_id, attrs):
        initial_data = db_manager._fetch_g_data()
        self.call(
            method_name="build_G_from_data",
            initial_data=initial_data,
            env_id=session_id,
            save_demo=True,
        )

"""


    def get_runtime_eq_params(
            self,
            params: List[str],
            attrs_struct: List[dict]
    ) -> dict:
        
        Get all runtime eq params for all attrs in the struct.
        Handles params from attrs, utils methods, and constants.
        Returns a dict mapping param keys to lists of values for each attr.
        
        Args:
            params: List of parameter keys to retrieve
            attrs_struct: List of attribute dictionaries for each node
            
        Returns:
            dict: Mapping of param_key -> list of values (one per attr in struct)
        

        data_soa = {}
        
        for key in params:
            vals = []
            
            # Try to get from attrs first
            for i, attrs in enumerate(attrs_struct):
                if key in attrs:
                    val = deserialize(attrs[key])
                    vals.append(val)
                else:
                    vals.append(None)
            
            # If any are None, try utils methods for those specific ones
            for i, attrs in enumerate(attrs_struct):
                if vals[i] is None:
                    try:
                        if hasattr(self, key):
                            val = self.get_step_eq_val(attrs, key)
                            if val is not None:
                                vals[i] = val
                    except Exception as e:
                        # If method exists but fails, will try constant lookup
                        pass
            
            # If still any None, try constants from UTILS_WORKER
            if any(v is None for v in vals):
                try:
                    uw = ray.get_actor(name="UTILS_WORKER")
                    node = ray.get(uw.get_node.remote(nid=key))
                    print(f"collect const {key}")
                    print("const n found", node)
                    if node and "value" in node:
                        # If constant found, use same value for all attrs that are None
                        const_val = node["value"]
                        print("value", const_val)
                        for i in range(len(vals)):
                            if vals[i] is None:
                                vals[i] = const_val
                    else:
                        # Constant node exists but no value - this is an error condition
                        raise ValueError(f"Constant {key} found in UTILS_WORKER but has no 'value' field")
                except Exception as e:
                    print(f"Error getting constant {key} from UTILS_WORKER: {e}")
                    # Re-raise if it was a ValueError about missing value
                    if isinstance(e, ValueError):
                        raise
            
            # Ensure we have the right number of values
            if len(vals) != len(attrs_struct):
                vals = [None] * len(attrs_struct)
            
            # Convert to array if all values are not None
            if all(v is not None for v in vals):
                try:
                    import jax.numpy as jnp
                    data_soa[key] = jnp.array(vals)
                except Exception:
                    data_soa[key] = vals
            else:
                # If some are None, try to handle gracefully
                data_soa[key] = vals
        
        return data_soa

"""

"""def neighbor_pm_val_same_type(self, attrs):
    ntype = attrs["type"]
    npm = attrs["npm"]
    #print(f"neighbor_pm_val_same_type for {ntype}")
    npmval = self.qfu.get_npm_values(
        npm=npm,
        ntype=ntype,
    )
    #print("npmval", npmval)
    return npmval
"""

"""    def neighbor_pm_val_fmunu(self, attrs):
    npm=attrs["npm"]
    type=attrs["type"]
    if type.lower() in G_FIELDS:
        npm_fmunu = self.qfu.get_npm_values(
            npm=npm,
            ntype=type,
            field_key="fmunu",
        )
        #print("neighbor_pm_val_fmunu item:", npm_fmunu)
        return npm_fmunu
    else:
        print("wrong ntype for neighbor_pm_val_fmunu:", type)
    return None
"""
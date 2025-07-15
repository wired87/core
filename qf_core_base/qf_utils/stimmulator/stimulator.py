import numpy as np

from qf_core_base.fermion.ferm_utils import FermUtils
from qf_core_base.qf_utils.field_utils import FieldUtils
from qf_core_base.qf_utils.stimmulator.stim_utils import StimUtils
from utils.graph.local_graph_utils import GUtils

import itertools

class Stimulator(StimUtils, FermUtils):

    """
    MVP / DEMO:
    - Superset photon G variations
    - loop
    - for n iters apply stim block X (each iter in block gets stimulus with slight changes)




    Stim process demo:
    - User writes stim command
    - AI first writes cfg file
    - lets the user check
    - Node attr val gets updated
    - pathway is triggered

    Update logik:
    MVP:
    bei jedem änderungswert falle zurück auf den
    anfang des lokalen pathway des feldes (zb elektron)

    todo mah einen ebsseren plan fürs stimulieren

    """

    def __init__(self, g: GUtils, types_of_interest, testing, demo):
        self.types_of_interest = types_of_interest
        self.g = g
        self.layer="STIMULATOR"
        self.nodes = list(attrs for _, attrs in self.g.G.nodes(data=True) if attrs.get("type") in self.types_of_interest)

        self.demo = True
        self.node_powerset = self._get_node_power_set()
        self.stim_queue = []
        self.total_runs=0
        self.field_utils = FieldUtils()
        self.loop=0
        self.stim_variant_step = 1000
        self.stim_increase = 1e9
        self.multiplier_step = None
        self.testing=testing
        self.demo=demo
        StimUtils.__init__(self, self.nodes)




    def select_nodes_in_area(self, center, radius=1):
        """
        As part of the sim area picker, get all nodes within a specific radius based on conter (start pos)
        :returns: List of node ids
        """
        return [n for n, attrs in self.g.G.nodes(data=True) if np.linalg.norm(np.array(n.pos[:2]) - np.array(center)) <= radius]


    def stim_from_pos(
            self,
            id_list: list,
            change_attrs=None,
    ):
        """
        time: todo update each nodes timer if
        all keys todo: prevalidate
        """

        qfns = [(k, v) for k, v in self.g.G.nodes(data=True) if k in id_list]
        for nid, attrs in qfns:
            if self.demo is True:
                attrs["energy"] += 1
            self.g.G.nodes(nid)["attrs"].update(change_attrs)
       #print("Stimuli applied!")



    def validate_stim_color(self, key):
        # run against possible_stm_types parent keys and choose color
        return (255,0,0,.2)


    def add_stimuli(self, stim_cfg):
        cat = stim_cfg["category"]
        pos = stim_cfg["pos"]
        radius = stim_cfg["radius"]
        duration = stim_cfg["duration"]

        # Set id
        stim_node_id = f"{cat}_{pos}_{radius}_{duration}"

        # Get overlapping neighbors
        nodes = self.select_nodes_in_area(center=pos, radius=radius)

        stim_cfg["color"] = self.validate_stim_color(key=stim_node_id)

        self.g.add_node(
            attrs=dict(
                id=stim_node_id,
                type=self.layer,
                **stim_cfg
            )
        )

        # connect stim to nodes
        for node_id in nodes:
            self.g.add_edge(
                stim_node_id,
                node_id,
                attrs=dict(
                    rel="stimulates",
                    src_layer=self.layer,
                    trgt_layer="QFN",
                )
            )



    def _get_node_power_set(self):
        """
        Erstellt ein Potenzset aller Knoten eines NetworkX-Graphen.
        Jedes Element im Potenzset ist ein frozenset der Knoten.
        """
        _set = [frozenset(subset) for r in range(len(self.nodes) + 1) for subset in itertools.combinations(self.nodes, r)]
       #print("_set created", _set)
        return _set


    def main(self):
        if self.demo is True:
            self._apply_stim()
        else:
            if self.total_runs > len(self.node_powerset):
                if self.loop >= self.stim_variant_step:
                    self._apply_stim()
                    self.loop = 0
                else:
                    self.loop += 1

                self.total_runs += 1
                return True
            else:
                return False
            

    def _apply_stim(self):
       #print("Apply Stimuli")
        stim_stage = self.node_powerset[0]
        for item in stim_stage:
            item_id = item["id"]
            ntype = item.get("type")

            # Apply stim PSI
            if ntype.lower() in self.field_utils.fermion_fields:
                attrs = self.g.G.nodes[item_id]
                self.g.update_node(attrs.update({
                    "psi": self.init_psi(
                        ntype,
                        serialize=True,
                        stim=True
                    )
                }))









    def update_stim(self, node_ids:list, env_attrs):
        def apply_stimuli(node_attrs, change_keys, sim_cfg):
            for change_key in change_keys:
                for k, v in self.possible_stm_types.items():
                    if change_key in v:
                        self.sl.log(f"Stim {change_key}")
                        if not isinstance(node_attrs[change_key], (int, float)):
                            try:
                                node_attrs[change_key] = float(node_attrs[change_key])
                            except Exception as e:
                               #print(f"Could not convert {change_key} to numeric:", e)
                                continue
                        if sim_cfg["type"] == "increase":
                            node_attrs[change_key] += sim_cfg["change"]
                        else:
                            node_attrs[change_key] -= sim_cfg["change"]
            return node_attrs

        all_stims = [(n, attrs) for n, attrs in self.g.G.nodes(data=True) if attrs.get("type") == self.layer]

        for node_id, stim_cfg in all_stims:
            neighbors = self.g.get_neighbor_list(node_id, target_type="QFN")
            for n in neighbors:
                neighbor_id = n[0]
                neighbor_attrs = n[1]
                change_keys = stim_cfg["change_keys"]

                # Update ndoe attrs
                neighbor_attrs = apply_stimuli(neighbor_attrs, change_keys, stim_cfg)

                # Save changes
                self.g.G.nodes[neighbor_id].update(neighbor_attrs)

                # Update sim_cfg -> todo move in TimeStepUpdator
                stim_cfg["time_step_progress"] += env_attrs["time_step"]
                if stim_cfg["time_step_progress"] < stim_cfg["duration"]:
                    self.g.G.nodes[node_id].update({
                        stim_cfg
                    })
                else:
                    self.g.remove_node(node_id, "QFN")





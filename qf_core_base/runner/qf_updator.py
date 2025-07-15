"""
Nein, nicht immer.
Aber:

✅ In vielen fundamentalen Fällen: Ja — das Higgs-Feld ist der erste Trigger.

Nachdem eqations fertig:
Higgs Stim:
Pattern (Lokalisierter Abstand)
Zeit
Shape
Amount

-> Umrchnen zu genauem typ von partikel (10 shocks in sector A = 1 Electron)

Genau das simmulieren wir


Statische und bewegliche teilchen im circuit


Mathe - Graph: modul L





Super feiner magnet mut super vielen stufen kann ein genaues circuit lokalisiert in die physische welt übertragen



Das hier ist alles nur ein training. Es ght nur darum das nächste level zu erreichen.
Alles ist nur eine projektion in deiner bubble -> du wirst bewertet

Die Energie die das HIGGS Feld anregt .ist was wir modeliern


"""
import time

from bm.settings import TEST_USER_ID
from itertools import chain, combinations

from qf_core_base.calculator.calculator import Calculator
from qf_core_base.fermion.ferm_base import FermionBase
from qf_core_base.g.GaugeBase import GaugeBase
from qf_core_base.higgs.higgs_base import HiggsBase
from qf_core_base.runner.qf_creator import QFCreator
from qf_core_base.qf_utils import QFLEXICON
from qf_core_base.qf_utils.all_subs import FERMIONS
from qf_core_base.qf_utils.mover import Mover
from qf_core_base.qf_utils.qf_utils import QFUtils

from utils.graph.local_graph_utils import GUtils
from utils.logger import LOGGER


def powerset_of_keysets(dict1, dict2):
    all_keys = list(dict1.keys()) + list(dict2.keys())
    return list(chain.from_iterable(combinations(all_keys, r) for r in range(len(all_keys) + 1)))



class QFUpdator(QFCreator):
    """
    This creates a graph of a lot of charged particles
    and let them interact following the core laws of physics

    How these particles are ordered and interact,
    this is the memory
    Like 8k forming a ring shape and create a field.

    Lagrangian let us dfine some classes with custommparameters (existing extt & customizable)

    Todo: jedes mal wenn ein update getriggert wird
    gibt der pathway einen payload über alle nodes mi


    Workflow:
    load_ray_remotes
    check hs + time.sleep - loop
    if all ready: send first timestep as trigger
    if not (after n tries): return problematic nodes (todo ai autonomous intervention)
    """

    def __init__(
            self,
            g: GUtils,
            env,
            testing=True,
            specs={},
            user_id=TEST_USER_ID,
            run=False,
    ):
        super().__init__(g, testing, specs)
        self.neighbors = None
        self.field_calculator = None
        # Core physics content
        self.user_id = user_id
        self.env=env
        self.g = g

        # Utility classes https://console.firebase.google.com/
        self.calculator = Calculator(
            g,
            equations=list(
                #STANDARD_MODELC
            )
        )

        self.qf_utils = QFUtils(
            g
        )

        self.loop=0
        self.run=run
        self.mover = Mover(g)
        self.qf_lex = QFLEXICON.copy()
        self.changepoints = []
        self.stack = {}
        #self.controller = Controller
        self.pool = {}

        LOGGER.info("QF Updator intitialisiert")


    def update_qfn(self):

        """
        1. Calc L for each
        2. Calc L against parent

        Synchron updates -> updates made within the loop just
        take affect within the next loop
        """
        LOGGER.info("Start update loop")

        stuff = [(nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if attrs.get("type") == "QFN"]
        len_stuff = len(stuff)
        index = 0
        #LOGGER.info(f"Startindex {index}, {len_stuff}")

        while index < len_stuff:
            # validate item
            updated_len_stuff = len(stuff)
            #LOGGER.info(f"updated_len_stuff: {updated_len_stuff}")

            if len_stuff < updated_len_stuff:
                #LOGGER.info("len_stuff < updated_len_stuff")
                index -= (updated_len_stuff - len_stuff)

            # Get data
            nid, attrs = stuff[index]
            #print("nattrs", attrs)
            self.update_process(
                self.env,
                attrs,
                len_stuff,
                nid,
            )
            index += 1


    def update_process(
            self,
            env_attrs,
            attrs,
            len_stuff,
            nid,
    ):
        # Entry: QFN
        # Option A: Calc everything, once per timestep
        # Cons: Overhead, not realistic
        # Pros: easy

        # Option B: Set Trigger which updates the pathway
        # Cons: Harder to implement (changes across QFN edges
        # Pros: highly realistic -> Go with it

        # -> Wir lauschen nicht nach attr changes sondern
        # neuen parametern im system -> interaktion triggert
        # pathway. ->
        # Erzeuge Künstliches Feld mit ... parametern
        # um Fermion, gauge usw zu triggern
       #print(f"Start UPDATE PROCESS")
        env_id = self.env.get("id")
        # GET self FIELDS (PSI etc.)
        if nid is None:
            nid = attrs.get("id")
        if env_id is None:
            env_id = attrs.get("id")
        if not env_id or not nid:
            LOGGER.info("bad request")
            return

        if attrs.get("neighbors_pm") is None:
            # Get neighbors in 3d space
            # enfachqfn ids müssen vorher gesetzt werden
            attrs["neighbors_pm"] = self.calculator.cutils.set_neighors_plus_minus(
                    node_id=nid,
                    self_attrs=attrs,
                    d=self.env["d_default"],
                    trgt_type="QFN",
                    field_type=attrs["parent"][0].lower()
                )

        ### VISUALS ###
        all_fields = self.qf_utils.get_all_node_sub_fields(nid)
        for fields_from_type in all_fields:
            for snid, sattrs in fields_from_type:
                self.update_core(
                    snid,
                    sattrs,
                    attrs,
                    nid,
                    env_attrs,
                    attrs["neighbors_pm"],
                )

                self._run_utils(snid, sattrs)


    def update_core(
            self,
            nid,
            attrs,
            qf_attrs,
            qf_nid,
            env_attrs,
            neighbors_pm,
    ):
        #print(f"workign {nid}")

        parents = [p.lower() for p in attrs.get("parent")]
        ntype = attrs.get("type")
        # print("PARENTS:", parents)

        attr_keys = [k for k in attrs.keys()]
        if "psi" in parents:
            sub_type = qf_attrs.get("sub_type")

            if sub_type == "ITEM":
                ferm_base = FermionBase(
                    self.g,
                    qfn_id=qf_nid,
                    nid=nid,
                    neighbors_pm=neighbors_pm,
                    d=self.env["d"],
                    attr_keys=attr_keys,
                    theta_W=env_attrs["theta_W"],
                    attrs=attrs,
                    **attrs
                )
                ferm_base.main()

            if sub_type == "BUCKET":
                # Todo outsrc tasks to fermion bucket (holds all three quark items) -> (like total quark lagrangian, delta_t vom triplet...)
                pass

        elif "phi" in parents:
            hu = HiggsBase(
                g=self.g,
                neighbors_pm=neighbors_pm,
                d=env_attrs["d"],
                attrs=attrs,
                env=self.env,
                **attrs
            )
            hu.main()
        elif "gauge" in parents and ntype.lower() != "gluon_bucket":
            gb = GaugeBase(
                g_utils=self.g,
                qfn_id=qf_nid,
                d=self.env["d"],
                attr_keys=attr_keys,
                neighbors_pm=neighbors_pm,
                attrs=attrs,
                env=self.env
            )
            gb.main()



    def event_sleep(self, type, len_stuff):
        if type in [f.upper() for f in FERMIONS]:
            if self.loop == ((len(FERMIONS) * len_stuff)+1):
                time.sleep(100)
            else:
                self.loop += 1
       #print("loop", self.loop)

    def update_single_field(self, nid, attrs, qfn_id, qfn_attrs, field_val_prev, parent):
       #print(f"Working field {nid}")
        # get all sub fields neihgbors
        attrs[parent] = self.calculator.cutils.dmu(
            nsum=qfn_attrs["neighbors_pm"],
            prev=field_val_prev,
            now=attrs[parent],  # psi
            d=self.env["d_default"],
            ntype=attrs.get("type"),
            parent=parent,
        )

    def _run_utils(self, nid, attrs):
        #print("Handle _run_utils")
        # update time

        # update G already in Field class ->
        # h entry in update G method
        #print(f"finished {nid}")
        attrs["time"] += 1















    def calc_field_sum(self, node_id, field_type):
        """
        Sum values of all neighbor fields of the same type.
        Stores the result in node[field_type.lower() + "_total"]
        """
        neighbors = self.g.get_neighbor_list(node_id, field_type)
        total = []
        for nid, attrs in neighbors:
            val = attrs.get(field_type.lower())
            if val is not None:
                total.append(val)
        return total


    def _laplacian_sum(self, neighbors):
        """
        Collect in all H values of all neighbors
        Used to calc
        """
        LOGGER.info("Set sum for laplacian")
        i = 0
        laplacian_sum = {}
        for neighbor_id, nattrs in neighbors:
            if i == 0:
                LOGGER.info("neighbor_id, nattrs", neighbor_id, nattrs)
            i +=1
            n_phi = nattrs.get("phi")
            laplacian_sum.update(n_phi)
        LOGGER.info("Finished laplacian")
        return laplacian_sum







    def _handle_edges(self, neighbors, attrs, nid, env_attrs):
        if neighbors:
            LOGGER.info("Working neighbors")
            for neighbor_id, nattrs in neighbors:
                attrs=self._handle_field_interaction(
                    attrs, nattrs, nid, neighbor_id, env_attrs
                )
        return attrs

    def _handle_field_interaction(self, attrs, nattrs, nid, nnid, env_attrs):
        # term value wird weitegereicht
        edge_attrs = self.g.G[nid][nnid]
        # get fermion params of th neighbor
        attrs=self.calculator.main(
            parent=attrs,
            env_attrs=env_attrs,
            child=nattrs,
            edge_attrs=None, # todo multigraph implementation -> multiple edges
            double=True,
            equations=[]#SM_INTERACTANT_EQS
        )
        return attrs

    def _distribute_vals(self, nattrs, attrs, neighbor_id):
        for k, v in self.qf_lex.items():
            # MEASUREMENTS COMMUNICATION
            if v["origin"] == "measured":
                pn = k
                n_val = nattrs[pn]["value"]
                self_val = attrs[pn]["value"]
                # todo make better!!!
                if isinstance(n_val, (int, float)) and isinstance(self_val, (int, float)) and self_val > n_val:
                    new_val = n_val + 0.5 * (self_val - n_val)
                    if new_val:
                        self.g.G.nodes[neighbor_id][pn] = new_val






"""

        >>>>>>>>>>>>>>>Calculate ALL sub-fields of a single qfn<<<<<<<<<<<<<<<<<<


    def update_all_fields(
            self,
            qfn_id,
            env_attrs,
            qfn_attrs,
    ):
        LOGGER.info(f"working: {qfn_id}")
        # GET self FIELDS (PSI etc.)
        (phi_id, phi_attrs), psis, gs = self.qf_utils.get_all_node_sub_fields(qfn_id)

        total_phi = []
        phi_self_neighbors = self.g.get_neighbor_list(phi_id, "PHI")

        #
        phi_attrs["nphi"] = self.calculator.cutils.set_neighors_plus_minus(
                    node_id=qfn_id,
                    self_attrs=qfn_attrs,
                    d=self.env["d_default"],
                    trgt_type="QFN",
                    field_type=qfn_attrs["parent"][0].lower()
                )

        # Calc self state
        self.calculator.main(
            parent=phi_attrs,  # E: is logged aS None
            parent_id=phi_id,
            env_attrs=env_attrs,
            child=None,
            edge_attrs=None,  # todo multigraph implementation -> multiple edges
            double=True,
            equations=PHI_EQ.copy()
        )

        # PHI -> PHI
        for pn in phi_self_neighbors:
            pn_id = pn[0]
            pn_attrs = pn[1]

            # Add to phi sum
            total_phi.append(pn_attrs.get("phi"))

            edge = self.g.G[phi_id][pn_id]

            #for index, edge in edges.items():
            self.calculator.main(
                parent=phi_attrs, # E: is logged aS None
                parent_id=phi_id,
                env_attrs=env_attrs,
                child=pn_attrs,
                edge_attrs=edge,  # todo multigraph implementation -> multiple edges
                double=True,
                equations=PHI_PHI_EQ.copy()
            )

        qfn_attrs["phi_total"] = total_phi

        # PHI -> GAUGE
        g_neighbors = self.g.get_neighbor_list(phi_id, "GAUGE")
        for g in g_neighbors:
            g_id = g[0]
            g_attrs = g[1]

            edge_attrs = self.g.G[phi_id][g_id]

            self.calculator.main(
                parent=phi_attrs,
                parent_id=g_id,
                env_attrs=env_attrs,
                child=g_attrs,
                edge_attrs=edge_attrs,  # todo multigraph implementation -> multiple edges
                double=True,
                equations=FERM_HIGGS_EQ.copy()
            )

        #################

        psi_sum = {}
        for psi_sub_field in psis:
            # E.g. Electron -> PHI
            psi_id = psi_sub_field[0]
            psi_attrs = psi_sub_field[1]
            psi_field_type = psi_sub_field.get("field")

            if psi_field_type not in psi_sum:
                qfn_attrs[psi_field_type] = []
            qfn_attrs[psi_field_type].append(psi_attrs.get("psi"))

            edge = self.g.G[psi_id][phi_id]  # no multigraph anymore

            # PSI -> PHI
            self.calculator.main(
                parent=psi_attrs,
                parent_id=psi_id,
                env_attrs=env_attrs,
                child=phi_attrs,
                edge_attrs=edge,  # todo multigraph implementation -> multiple edges
                double=True,
                equations=FERM_HIGGS_EQ.copy()
            )

            # todo L_y >= mass = particle expression -> reset L_y = 0 (+ diference) psi*=.1
            mass = edge["mass"]
            L_y = edge["L_y"]
            edge_type = edge["type"]
            if L_y >= mass:
                # Express particle

                self.g.add_node(
                    dict(
                        id=f"{edge_type.upper()}_{len([k for k, v in self.g.G.nodes(data=True) if v.get('type') == edge_type.upper()]) + 1}"
                        # todo particle params
                    )
                )
                # todo https://docs.google.com/spreadsheets/d/1qqTbK8nugFftRFt5z5CtoGEkj1rTv9cmT4vcr3LxUAE/edit?gid=0#gid=0&range=A100
                # reset fields
                edge["psi"] *= .1
                edge["L_y"] = 0

            # PSI -> GAUGE (intern & extern)
            gauge_neighbors = self.g.get_neighbor_list(psi_id, "GAUGE")
            for g in gauge_neighbors:
                g_id = g[0]
                g_attrs = g[1]
                self.calculator.main(
                    parent_id=psi_id,
                    parent=psi_attrs,
                    env_attrs=env_attrs,
                    child=g_attrs,
                    edge_attrs=None,
                    double=True,
                    equations=FERM_GAUGE_EQ.copy()
                )

        # GAUGE -> GAUGE
        gauge_sum ={}
        for g in gs:
            g_id = g[0]
            g_attrs = g[1]
            gauge_field = g_attrs.get("field")

            self.g_neighbors = self.g.get_neighbor_list(g_id, "GAUGE")

            for n in self.g_neighbors:
                neighbor_id = n[0]
                neighbor_attrs = n[1]

                # Handle sum coupling consatants neighbors
                if g not in gauge_sum:
                    qfn_attrs[gauge_field] = []
                qfn_attrs[gauge_field].append(neighbor_attrs.get("A_mu"))

                # Run G -> G equations
                edge_attrs = self.g.G[g_id][neighbor_id]

                self.calculator.main(
                    parent_id=g_id,
                    parent=g_attrs,
                    env_attrs=env_attrs,
                    child=neighbor_attrs,
                    edge_attrs=edge_attrs,
                    double=True,
                    equations=FERM_GAUGE_EQ.copy()
                )


        return qfn_attrs




"""
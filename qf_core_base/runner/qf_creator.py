"""
🔁 Beispiel-Kopplungstabelle:

Quelle	Ziel	Verbindung über Nodes hinweg erlaubt?	Grund
PSI_A	PHI_B	❌	Masse ist lokal
PHI_A	PHI_B	✅	Feldgradient
GAUGE_A	GAUGE_B	✅	Kraftvermittlung
PSI_A	PSI_B	⚠️ optional	Nur bei Bewegung / Transport
EQUATIONS CHANGE IN LIFECYCLE
INDIVIDUALLY

CLASSIFY VARS INTO:
kinetic_terms → movement
potential_terms → self-interaction / mass / shape
interaction_terms → external couplings, field-to-field
gauge_terms → EM fields, vector bosons, etc.
topological_terms, boundary_terms, etc. (for more advanced cases)
CAN THEN FREELY CHOOSEN TO DEFINE THE PROPERTIES OF THE SYS

Automatic derivation engines (like FeynRules, PyR@TE)

Do it now without Graph -> add a graph when the system changes
Categorize params
link to possible equations#

todo graph of single equations: from qf_sim.calculator.knowledgebase import EQPathwayCreator
"""

import numpy as np

from qf_core_base.calculator.calculator import Calculator
from qf_core_base.fermion.ferm_creator import FermCreator
from qf_core_base.fermion.ferm_utils import FermUtils
from qf_core_base.g.g_creator import GaugeCreator
from qf_core_base.g.gauge_utils import GaugeUtils
from qf_core_base.higgs.phi_utils import HiggsUtils

from qf_core_base.qf_utils.qf_utils import QFUtils
from qf_core_base.qf_utils.mover import Mover
from qf_core_base.qf_utils.field_utils import FieldUtils

from utils._np.serialize_complex import serialize_complex
from utils.graph.local_graph_utils import GUtils

# Params
from qf_core_base.higgs import HIGGS_PARAMS
from qf_core_base.fermion import FERM_PARAMS, PSI_UNIFORM


class QFCreator:
    """
    Get 3d matrix of qfns.
    0. Create PIXEL (Like a super Quantum Field)
    1. Create for each pixel all its sub fields
    2. Set Pos
    3. Link PIXEL neighbors
    4. Link single sub-fields

    ### @ Relay ###
    todo nach pause
    5. Create Equations
    6. Link source fields together
    """

    def __init__(self, g, testing, dim, **args):
        super().__init__()
        self.calculator = Calculator()
        self.field_utils = FieldUtils()
        self.fu = FermUtils()
        self.hu = HiggsUtils()
        self.gu = GaugeUtils()

        self.dim = None
        self.layer = "PIXEL"

        self.g = g  # GU -> struct
        self.mover = Mover(g)

        self.testing = testing
        self.dim = dim
        self.save_path= r"/qf_core_base/calculator\tree.json"

        # Helper
        self.qf_utils = QFUtils(g)

        self.g_creator = GaugeCreator(
            g_utils=self.g,
        )

        self.ferm_creator = FermCreator(
            g=self.g,
        )

        self.testing = True
        # todo SUPERSYMETRY



    def create(self, cluster_dim, env):
        self.env = env
        print("QF.create")
        self.dim = cluster_dim
        # Add QQ
        amount_needed_nodes = self.get_amount_nodes()

        for i in range(amount_needed_nodes):
            self.create_single_pixel(index=i, env_id=env["id"])

        # 2. - 5.
        self.spread_connect_qfn()
        self.g.print_edges("GLUON", "GLUON")

        #self.g.print_status_G()
        #self.g.save_graph(dest_name=self.save_path)

        print("Creation finished")
        #time.sleep(100)

    def create_single_pixel(self, index, env_id):
        nid = f"px_{index}"
        ##print("Creating PIXEL", nid)
        self.g.add_node(
            {
                "id": nid,
                "parent": ["ENV"],
                "L_total": 0.0,
                "type": "PIXEL",
                "pos": [0.0, 0.0, 0.0]
                #**{k: v["value"] for k, v in self.qf_lex.items() if "value" in v},
            }
        )

        print("PIXEL created:", nid)

        # Link PIXEL to QF
        self.g.add_edge(
            env_id,
            nid,
            attrs=dict(
                rel="has_point",
                src_layer="ENV",
                trgt_layer="PIXEL",
            )
        )

        # Create Single fields
        # todo create nodes from single key - value pairs
        phi_id = self.higgs_creator(src_qfn_id=nid)
        self.ferm_creator.create(src_qfn_id=nid)
        self.g_creator.create(src_qfn_id=nid)

        # Connect local nodes
        self._connect_local_fields(nid)

        self.g.print_edges("GLUON", "GLUON")


    def connect_field_types(
            self,
            src_qfn_id,
            trgt_qfn_id=None,
    ):
        """
        Connects all fields to valid neighbors (either local (same node) or second node)
        Use for neighbor connection
        """
        # todo stell jedes feld auf einen ode um! komplexität des graphen egal ->
        # du multitreadest das eh alles

        ##print("Connect field types")
        src_fields = None

        if trgt_qfn_id is None:
            # Connection within same PIXEL
            trgt_qfn_id = src_qfn_id

        if trgt_qfn_id != src_qfn_id:
            # Get all fields of th trgt node
            src_fields:dict = self.qf_utils.get_all_node_sub_fields(nid=src_qfn_id)

        # Get all fields of th trgt node
        trgt_fields:dict = self.qf_utils.get_all_node_sub_fields(nid=trgt_qfn_id)

        # HIGGS #####################################
        self._connect_phi(
            src_fields=src_fields["PHI"] if src_fields is not None else trgt_fields["PHI"],
            trgt_phi=trgt_fields["PHI"],
            src_qfn_id=src_qfn_id,
            trgt_qfn_id=trgt_qfn_id,
        )

        # FERMION ###############################
        # Does not couple across nodes

        # GAUGE ###############################
        self._connect_gauge(
            gauge_fields=src_fields["GAUGE"] if src_fields is not None else trgt_fields["GAUGE"],
            trgt_gs=trgt_fields["GAUGE"],
            trgt_ferms=trgt_fields["FERMION"],
            src_qfn_id=src_qfn_id,
            trgt_qfn_id=trgt_qfn_id,
        )
        print("Finished connect field types")

    def _connect_phi(
            self,
            trgt_phi,
            src_fields,
            src_qfn_id,
            trgt_qfn_id

    ):
        k = "PHI"
        print("Conect HIGGS")
        src_phi_id = list(src_fields.keys())[0] # src_fields[0][0]
        if trgt_qfn_id != src_qfn_id:
            # Higgs -> Higgs
            trgt_phi_id = list(trgt_phi.keys())[0]
            self.g.add_edge(
                src_phi_id,
                trgt_phi_id,
                attrs=dict(
                    rel="extern_coupling",
                    src_layer=k.upper(),
                    trgt_layer=k.upper(),
                    time=0.0,
                )
            )


    def _connect_gauge(
            self,
            trgt_gs,
            gauge_fields,
            src_qfn_id,
            trgt_qfn_id,
            trgt_ferms
    ):
        k = "GAUGE"
        print("_connect_gauge")
        for g_field_id, g_field_attrs in gauge_fields.items():
            g_field_type = g_field_attrs.get("type")

            # Get valid coupling partners
            fields_to_couple:list = self.field_utils.gauge_to_gauge_couplings[g_field_type.lower()]

            # 1. Gauge ➝ Gauge INTERN & EXTERN
            for trgt_field_id, trgt_field_attrs in trgt_gs.items():
                trgt_field_type = trgt_field_attrs.get("field")
                # Valid coupling partner?
                if trgt_field_type in fields_to_couple:
                    if src_qfn_id != trgt_qfn_id:
                        if g_field_type == "photon" and trgt_field_type == "photon":
                            continue
                        # Photon koppelt nicht an sich selbst im klassischen Sinn
                        # todo -> in welchem sinn sonst?
                        self.g.add_edge(
                            g_field_id,
                            trgt_field_id,
                            attrs=dict(
                                rel=f"extern_coupling",
                                src_layer=k,
                                trgt_layer=k,
                                #**GAUGE_PARAMS,
                            )
                        )

            # GLUON_ITEM -> GLUON_ITEM INTERN & EXTERN
            if g_field_type.upper() == "GLUON":
                # Create Gluon Gluon
                self.g_creator.connect_gluons(nid=g_field_id)


            # G -> FERM accross nodes
            # (FERM -> G -> FERM - interaction)
            gauge_ferm_coupling_partners:list = self.field_utils.gauge_to_fermion_couplings[g_field_type.lower()]
            for fermid, ferm_attrs in trgt_ferms.items():
                ferm_type = ferm_attrs.get("type")
                for partner in gauge_ferm_coupling_partners:
                    if partner in fermid:
                        self.g.add_edge(
                            g_field_id,
                            fermid,
                            attrs=dict(
                                rel=f"extern_coupling",
                                src_layer=g_field_type,
                                trgt_layer=ferm_type,
                            )
                        )
                        break


    def get_amount_nodes(self):
        print("dim:", self.dim)
        if self.dim:
            amount_nodes = 1
            for d in self.dim:
                amount_nodes *= d
            print("Create amount nodes", amount_nodes)
            # time.sleep(5)
            return amount_nodes



    ############################################################################



    def higgs_creator(self, src_qfn_id):
        phi = f"PHI"
        field_id = f"phi_{src_qfn_id}"
        h = 0.0  # repräsentiert die flukation um den vacuumswert (vev + h)
        print("Create Node", field_id)
        self.g.add_node(
            dict(
                id=field_id,
                type=phi,
                #_symmetry_groups=self.field_utils.get_sym_group(phi),
                time=0.0,
                d_phi=self.hu.init_d_phi(),
                h_prev=h,
                h=h,
                phi=self.hu.init_phi(h),
                **HIGGS_PARAMS,
            )
        )
        # Connect PHI to PIXEL
        self.g.add_edge(
            src_qfn_id,
            field_id,
            attrs=dict(
                rel=f"has_field",
                src_layer=self.layer,
                trgt_layer=phi,
            )
        )

       #print("Phi created")
        return field_id

    def psi_creator(self, src_qfn_id):
        # PSI
        for ferm_field, attrs in FERM_PARAMS.items():
            field_id = f"{ferm_field}_{src_qfn_id}"

            psi = self.fu.init_psi(ntype=ferm_field, serialize=False)
            psi_bar = self.fu._psi_bar(psi, self.fu._is_quark(ferm_field))

            ##print("Create Node", field_id)
            self.g.add_node(
                dict(
                    id=field_id,
                    type=ferm_field,
                    parent=["PSI"],
                    time=0.0,
                    psi=serialize_complex(com=psi),
                    psi_bar=serialize_complex(com=psi_bar),
                    #_symmetry_groups=self.fu.get_sym_group(ferm_field),
                    **PSI_UNIFORM,
                    **attrs,
                )
            )

            # PSI -> PIXEL
            self.g.add_edge(
                src_qfn_id,
                f"{field_id}",
                attrs=dict(
                    rel=f"has_field",
                    src_layer=self.layer,
                    trgt_layer=ferm_field,
                )
            )
            # psi does not self couple across PIXEL edges
            # Create & Connect params
            """self._create_connect_params_to_fields(
                field_id=field_id,
                param_struct=dict(
                    **PSI_UNIFORM,
                    **attrs,
                ),
                parent_layer=ferm_field,
            )"""
       #print("Psi created")



    def _create_connect_params_to_fields(
            self,
            field_id,
            param_struct, #PHI_PARAMS
            parent_layer, # "PHI"
    ):

        # CREATE PARAM TREE
       #print("Connect Params to", field_id, parent_layer)
        for k, v in param_struct.items():
            pid = f"{k}_{field_id}"
            self.g.add_node(
                dict(
                    id=pid,
                    type=k,
                    value=v,
                    parent=[parent_layer],
                )
            )
            # Connect PARAM to PARENT
            self.g.add_edge(
                field_id,
                pid,
                attrs=dict(
                    rel=f"has_param",
                    src_layer=parent_layer,
                    trgt_layer=k,
                )
            )




    # problem = types
    def _connect_local_fields(self, nid):
        print("Connect local fields")
        all_subs = self.qf_utils.get_all_node_sub_fields(nid)

        phi_id = list(all_subs['PHI'].keys())[0]
        phi_attrs = list(all_subs['PHI'].values())[0]

        print(f"PHI data extracted: {phi_id}: {phi_attrs}")

        for gid, gattrs in all_subs["GAUGE"].items():
            #print(" gid, gattrs",  gid, gattrs)
            # G -> PHI
            g_type = gattrs.get("type")
            self.g.add_edge(
                gid,
                f"{phi_id}",
                attrs=dict(
                    rel=f"intern_coupling",
                    src_layer=g_type.upper(),
                    trgt_layer="PHI",
                )
            )

        for fid, fattrs in all_subs["FERMION"].items():
            # Ferms -> PSIS
            ftype= fattrs.get("type")
            self.g.add_edge(
                fid,
                f"{phi_id}",
                attrs=dict(
                    rel=f"intern_coupling",
                    src_layer=ftype.upper(),
                    trgt_layer="PHI",

                )
            )

            # Ferm -> G
            for g in self.field_utils.fermion_to_gauge_couplings[ftype.lower()]:
                if g == "gluon":
                    partners = self.g.get_neighbor_list(nid, g.upper())
                    for partner_id, pattrs in partners.items():
                        self.g.add_edge(
                            fid,
                            partner_id,
                            attrs=dict(
                                rel=f"intern_coupling",
                                src_layer=ftype.upper(),
                                trgt_layer=g.upper(),
                            )
                        )
                else:
                    partner_id, pattrs = self.g.get_single_neighbor_nx(nid, g.upper())
                    #print(f"{fid} -> ", partner_id, g.upper())
                    self.g.add_edge(
                        fid,
                        partner_id,
                        attrs=dict(
                            rel=f"intern_coupling",
                            src_layer=ftype.upper(),
                            trgt_layer=g.upper(),
                        )
                    )
        print("Local field connection finished")


    def spread_connect_qfn(self):
        print("Spread and connect subs accross nodes...")

        # 1. iter
        # SPREAD ITEMS OVER VIRTUAL AREA
        spread_items = [
            (nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if
            attrs.get("type") == self.layer
        ]

       #print(f"Spread {len(spread_items)} items")
        # räumlicher abstand überall gelich (nur Zeit anders)
        d = self.env["d_default"]

        for index, (nid, attrs) in enumerate(spread_items):
            ##print("Dpread item", nid)
            attrs = self.mover.spread_objects_3d(
                amount_items=len(spread_items),
                dim=self.dim[0],  # x,y,z
                self_attrs=attrs,
                spread_evenly=d
            )
            print(f"{nid} pos: {attrs.get('pos')}")

            self.g.update_node(attrs)

            # Scatter sub-fields around pixel center pos
            self.scatter_nodes_near_position(
                qfn_id=nid,
                center_pos=attrs["pos"],
            )

        # 2. Iter
        # Connect nearest PIXEL neighbors
        spread_items = [
            (nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if
            attrs.get("type") == self.layer
        ]

        for nid, attrs in spread_items:
            attrs = self.set_neighbors(
                nid,
                d=d,
                self_attrs=attrs,
                all_nodes=spread_items
            )
            self.g.update_node(attrs)

        # 3. Iter
        # Connect sub fields across PIXELs
        self.connect_subs_across_qfn()

        # 4. Connect Params between all sub-fields (intern & extern)
        # -> fallback to equaton linking
        # -> finish here - the calculator gets alltimes build up at sim start

        self.mover.distribute_subpoints_around_qfns()
        print("Spread finished")

    def connect_subs_across_qfn(self):
        spread_items = [
            (nid, attrs) for nid, attrs in self.g.G.nodes(data=True) if
            attrs.get("type") == self.layer
        ]
        for nid, attrs in spread_items:
            neighbors = self.g.get_neighbor_list(nid, "PIXEL")
            for n in neighbors.items():
                # Connect all fields direct
                self.connect_field_types(
                    src_qfn_id=nid,
                    trgt_qfn_id=n[0]
                )


    def scatter_nodes_near_position(self, qfn_id, center_pos, radius=20):
        intern_neighbors = self.g.get_neighbor_list_rel(qfn_id, trgt_rel="has_field")
        center_pos = np.array(center_pos)

        for nid, _ in intern_neighbors:
            # Kugelförmig gleichverteiltes Offset erzeugen
            direction = np.random.normal(size=3)
            direction /= np.linalg.norm(direction)  # Einheitsvektor
            distance = radius * np.random.rand() ** (1 / 3)
            offset = direction * distance

            new_pos = center_pos + offset
           #print("CENTER POS:", center_pos, "Offset:", offset, "New Pos:", new_pos)

            self.g.update_node({"id": nid, "pos": new_pos.tolist()})

    def set_neighbors(self, nid, self_attrs, d: int, all_nodes):
       #print("Set neighbors")
        #print("d", d)
        # d = distance
        nsum_phi = {}

        direction_definitions = {
            "x": (1, 0, 0),
            "y": (0, 1, 0),
            "z": (0, 0, 1),

        }
        """"xy_pp": (1, 1, 0),
                    "xy_pm": (1, -1, 0),
                    "xz_pp": (1, 0, 1),
                    "xz_pm": (1, 0, -1),
                    "yz_pp": (0, 1, 1),
                    "yz_pm": (0, 1, -1),
                    "xyz_ppp": (1, 1, 1),
                    "xyz_ppm": (1, 1, -1),
                    "xyz_pmp": (1, -1, 1),
                    "xyz_pmm": (1, -1, -1)
                    """
        self_pos = np.array(self_attrs["pos"])

        node_pos_dict = {node: np.array(attrs.get("pos")) for node, attrs in all_nodes}

        for direction_name, direction_matrix in direction_definitions.items():
            offset = np.array(direction_matrix) * d
            pos_plus = self_pos + offset
            pos_minus = self_pos - offset

            node_plus=None
            node_minus=None

            # get id form node
            for k, v in node_pos_dict.items():
                #print("Compare Ppos:", v, "with", pos_plus)
                if np.allclose(v, pos_plus, atol=1e-6):
                    node_plus = k

            for k, v in node_pos_dict.items():
                #print("Compare Mpos:", v, "with", pos_minus)
                if np.allclose(v, pos_minus, atol=1e-6):
                    node_minus = k


            if not node_plus:
                node_plus = nid

            if not node_minus:
                node_minus = nid


            nsum_phi.update(
                {
                    direction_name: [
                        node_plus,  # + npsi
                        node_minus,  # - npsi
                    ]
                }
            )

            if node_plus and node_plus != nid:
                #print(f"Connect {nid}->{node_plus}")
                self.g.add_edge(
                    nid,
                    node_plus,
                    attrs=dict(
                        trgt_layer=self.layer,
                        src_layer=self.layer,
                        rel="neighbor"
                    )
                )

            if node_minus and node_minus != nid:
               #print(f"Connect {nid}->{node_minus}")
                self.g.add_edge(
                    nid,
                    node_minus,
                    attrs=dict(
                        trgt_layer=self.layer,
                        src_layer=self.layer,
                        rel="neighbor"
                    )
                )

           #print(f"Connected {nid} -> {node_minus}-{node_pos_dict[node_minus] if node_minus else None} & {node_plus}-{node_pos_dict[node_plus] if node_plus else None}")

            self_attrs["neighbors_pm"] = nsum_phi

        return self_attrs


if __name__ == "__main__":
    image_path = r"C:\Users\wired\OneDrive\Desktop\Projects\Brainmaster\physics\quantum_fields\lexicon\lex_graph.html"
    qfc = QFCreator(
        g=GUtils(),
        testing=True,
        dim={
            "shape": "rect",
            "dim": [2, 2, 2]
        }
    )





"""

"xy_pp": (1, 1, 0),
            "xy_pm": (1, -1, 0),
            "xz_pp": (1, 0, 1),
            "xz_pm": (1, 0, -1),
            "yz_pp": (0, 1, 1),
            "yz_pm": (0, 1, -1),
            "xyz_ppp": (1, 1, 1),
            "xyz_ppm": (1, 1, -1),
            "xyz_pmp": (1, -1, 1),
            "xyz_pmm": (1, -1, -1)
    def handle_fields(self, qfn_id):
        # Create a node for Each field type
        self.create_fields(src_qfn_id=qfn_id)
        # Connect sub field types (electron etc)
        self.connect_field_types(qfn_id)

    def create_fields(self, src_qfn_id):

        TODO:
        ERSTELL DIESEN GRAPHEN NUR EIN MAL
        LADE ALLE GLEICHUNGEN GENAU HIER MIT
        REIN. Kalkulator nimmt jedes mal
        einfach die werte der gegebenen nodes
        aus dem env (anderer G)
        Creates all types of quantum fields
         Primary use for collecting edge attrs
        # -> ["psi", "phi", "gauge"]todo handle more fields later

        for k in ["psi", "phi", "gauge"]:
            src_id = f"{k.upper()}_{src_qfn_id}"
            if k.lower() == "psi":
                # PSI
                for ferm_field in self.fermion_fields:
                    field_id = f"{ferm_field}{src_qfn_id}"
                    ##print("Create Node", field_id)
                    self.g.add_node(
                        dict(
                            id=field_id,
                            type=k.upper(),
                            field=ferm_field,
                            **FERM_PARAMS[ferm_field],
                        )
                    )

                    # Connect to PIXEL
                    self.g.add_edge(
                        src_qfn_id,
                        f"{field_id}",
                        attrs=dict(
                            rel=f"has_field",
                            src_layer="PIXEL",
                            trgt_layer=k.upper(),
                        )
                    )
            if k.lower() == "phi":
                field_id = f"phi{src_qfn_id}"
                ##print("Create Node", field_id)
                self.g.add_node(
                    dict(
                        id=field_id,
                        type=f"PHI",
                        **PHI_PARAMS,
                    )
                )
                # Connect to PIXEL
                self.g.add_edge(
                    src_qfn_id,
                    f"{field_id}",
                    attrs=dict(
                        rel=f"has_field",
                        src_layer="PIXEL",
                        trgt_layer=k.upper(),
                    )
                )

            elif k.lower() == "gauge":
                for gauge in self.gauge_fields:
                    field_id = f"{gauge}{src_qfn_id}"
                    ##print("Create Node", field_id)
                    self.g.add_node(
                        dict(
                            id=field_id,
                            type=k.upper(),
                            field=gauge,
                            **GAUGE_PARAMS,
                        )
                    )
                    # Connect to PIXEL
                    self.g.add_edge(
                        src_qfn_id,
                        f"{field_id}",
                        attrs=dict(
                            rel=f"has_field",
                            src_layer="PIXEL",
                            trgt_layer=k.upper(),
                        )
                    )

        # todo get center node
        # debug pos set  -> look 3d_pos def
        ## Get center node from defined pos
        for nid, attrs in spread_items:
            check_center:bool = self.calculator.set_neighors_plus_minus(
                node_id=nid,
                self_attrs=attrs,
                d=d,
                all_pixel_nodes=spread_items,
                check_center=True
            )
            if check_center is True:
                attrs.update({"center": True})
                print("center node set", nid)
                self.g.update_node(attrs)
            else:
                raise ValueError("Couldnt set ceneter node")


    def g_creator(self, src_qfn_id):
        # todo create nodes of all sngle vecs -> same with quarks

        g = "GAUGE"
        gluon = "GLUON"

        for g_field, gattrs in GAUGE_FIELDS.items():
            # Create Parent
            parent_attrs = self._create_g_parent(
                g_field,
                src_qfn_id,
                g,
                gattrs
            )

            # Define (sub) fields
            if g_field.lower() == "gluon":
                self._create_gluon_items(
                    parent_attrs["id"],
                    g_field,
                    gattrs,
                    src_qfn_id,
                    g,
                    gluon,
                )

       #print("G created")


    def _get_gauge_base_payload(self, g_field):
        # Field Value
        field_value = self.gu.init_G(ntype=g_field)

        # Set field value
        field_key = self.gu._field_value(g_field)

        dmu_field_key = self.gu.init_fmunu(ntype=g_field)

        fmunu = self.gu.init_fmunu(ntype=g_field)

        j_nu = self.gu.init_j_nu(ntype=g_field)

        return field_key, field_value, dmu_field_key, fmunu, j_nu
    def _create_gluon_items(
            self,
            parent_id,
            g_field,
            gattrs,
            src_qfn_id,
            g,
            gluon,

    ):

        for i in range(8):
            gauge_id = f"{g_field}_{i}_{src_qfn_id}"
            g_item_field = f"{g_field}_{i}"
            field_key, field_value, dmu_field_key, fmunu, j_nu = self._get_gauge_base_payload(
                g_item_field
            )
            attrs = dict(
                id=gauge_id,
                parent=[g, gluon],
                time=0.0,
                type=g_item_field,
                j_nu=j_nu,
                F_mu_nu=fmunu,
                **gattrs,
            )

            attrs[field_key] = field_value[i]
            attrs[f"dmu_{field_key}"] = dmu_field_key[i]

            self.g.add_node(attrs=attrs)

            # G_ITEM -> GLUON
            self.g.add_edge(
                parent_id,
                gauge_id,
                attrs=dict(
                    rel=f"has_instance",
                    src_layer=g,
                    trgt_layer=g_item_field,
                )
            )
       #print("Gluon items created")


    def _create_g_parent(
            self,
            g_field,
            src_qfn_id,
            g,
            attrs
    ):
       #print("Create parent G field")
        symmetry_groups = self.gu.get_sym_group(g_field)

        gauge_id = f"{g_field}_{src_qfn_id}"
        if g_field == "gluon":
            parent_attrs = dict(
                id=gauge_id,
                type=g_field,
                _symmetry_groups=symmetry_groups,
                f_abc=None,
                time=0.0,
            )
        else:
            field_key, field_value, dmu_field_key, fmunu, j_nu = self._get_gauge_base_payload(
                g_field
            )
            parent_attrs = dict(
                id=gauge_id,
                parent=[g],
                time=0.0,
                type=g_field,
                j_nu=j_nu,
                F_mu_nu=fmunu,
                f_abc=None,
                **attrs,
            )
            attrs[field_key] = field_value
            attrs[f"dmu_{field_key}"] = dmu_field_key

        # Create Gauge Node
        self.g.add_node(
            attrs=parent_attrs
        )

        # G -> PIXEL
        self.g.add_edge(
            src_qfn_id,
            f"{gauge_id}",
            attrs=dict(
                rel=f"has_field",
                src_layer=self.layer,
                trgt_layer=g_field,
            )
        )
        return parent_attrs
"""
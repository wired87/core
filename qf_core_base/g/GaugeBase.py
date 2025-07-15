import numpy as np

from qf_core_base.calculator.calculator import Calculator
from qf_core_base.g.gauge_utils import GaugeUtils
from qf_core_base.g.vertex import Vertex
from qf_core_base.qf_utils.all_subs import FERMIONS
from qf_core_base.qf_utils.qf_utils import QFUtils
from qf_core_base.symmetry_goups.main import SymMain
from utils._np.serialize_complex import deserialize_complex, check_serialize_dict
from utils.graph.local_graph_utils import GUtils
from itertools import product


class GaugeBase(GaugeUtils):

    """
    Repräsentiert ein qf_core_base / universelles Gauge feld welches durch
    erweiterte isntanzen angepasst werden kann


    Funktionen defineiren:
    Kinetic-Term
    Feldstärketensor


    Kopplungen
    H -> G

    🔹 2️⃣ Eichfelder (Gauge Fields)

    Das sind Wechselwirkungsfelder (Photon, W, Z, Gluonen).

    Sie vermitteln die Kräfte.

    Ohne sie keine Bindung, keine Dynamik, keine Ladung.

    ✅ Gauge-Felder sind fundamental als Vermittler aller Wechselwirkungen.
    Was du für Eichfelder brauchst:
    Eichfelder sind komplexer als skalare (Higgs) oder fermionische Felder. Sie beschreiben Kräfte und haben eine eigene innere Struktur.

    Feldwert verstehen (der Eichfeld-Wert A_
    mu):

    Im Gegensatz zum skalaren Higgs-Feld ist ein Eichfeld ein Vektorfeld. Es hat nicht nur einen Wert an jedem Punkt, sondern auch eine Richtung und eine "Farbe" oder "Ladung" im inneren Symmetrieraum.

    Für die elektromagnetische Kraft ist es das Photonenfeld (A_
    mu). Es hat 4 Komponenten für die Raumzeit (A_t,A_x,A_y,A_z).

    Für die schwache Kraft sind es die W- und Z-Bosonen. Diese haben nicht nur 4 Raumzeit-Komponenten, sondern auch noch eine zusätzliche "schwache Isospin"-Struktur. Die W-Bosonen sind geladen (W
    +
     ,W
    −
     ), das Z-Boson ist neutral (Z
    0
     ).

    Für die starke Kraft (QCD) sind es die Gluonen (A_
    mu
    a
     ). Sie haben ebenfalls 4 Raumzeit-Komponenten und zusätzlich 8 "Farben"-Indizes (von a=1 bis 8).

    Zusammensetzung des Feldwertes (Numerisches Beispiel für Photonenfeld):

    Nehmen wir das einfachste Eichfeld, das Photonenfeld (A_
    mu), das die elektromagnetische Kraft vermittelt. An jedem Punkt im Raumzeit-Gitter hat es einen Wert, der ein 4-Vektor ist.

    Numerisch würde ein Feldwert an einem Gitterpunkt so aussehen:

    # field_value an einem Punkt (t, x, y, z)
    photon_field_value = np.array([
        [A_t],  # Zeit-Komponente (oft elektrisches Potenzial)
        [A_x],  # Raum-Komponente x (Magnetisches Vektorpotenzial x)
        [A_y],  # Raum-Komponente y (Magnetisches Vektorpotenzial y)
        [A_z]   # Raum-Komponente z (Magnetisches Vektorpotenzial z)
    ], dtype=complex) # Kann komplex sein, aber oft reell für reale Felder
    Für ein Gluonenfeld hättest du 8 solcher 4-Vektoren an jedem Punkt.

    Für W/Z-Bosonen ist es noch komplexer wegen ihrer geladenen Natur und der Symmetrie.

    Was zu Veränderungen im Feld führt:

    Quellen: Eichfelder ändern sich durch das Vorhandensein von Ladungen.

    Das Photonenfeld ändert sich durch elektrische Ladungen (z.B. Elektronen, Quarks).

    Die W- und Z-Felder ändern sich durch schwache Ladungen (alle Fermionen haben schwache Ladung).

    Die Gluonenfelder ändern sich durch Farbladungen (Quarks und die Gluonen selbst, da Gluonen auch Farbladung tragen!).

    Selbstwechselwirkung (für W/Z und Gluonen): Im Gegensatz zu Photonen (die keine Selbstwechselwirkung haben), können W-, Z- und Gluonenfelder mit sich selbst wechselwirken. Das macht ihre Dynamik viel komplizierter.

    Higgs-Feld (für W und Z): Für W- und Z-Bosonen führt die Wechselwirkung mit dem Higgs-Feld (nach der Symmetriebrechung) zu ihrer Masse.

    Was verändert wird:

    Die Komponenten des Eichfeld-Vektors (A_t,A_x,A_y,A_z) an jedem Punkt werden über die Zeit hinweg aktualisiert. Ihre Dynamik wird durch die Maxwell-Gleichungen (für Photonen) oder die Yang-Mills-Gleichungen (für W/Z und Gluonen) beschrieben.

    Auswirkungen auf andere Systeme:

    Kraftübertragung: Eichfelder sind die Kraftträger. Ihre Anwesenheit und Veränderung führt zu Kräften, die auf geladene Teilchen wirken (z.B. ein Elektron wird vom Photonenfeld angezogen/abgestoßen).

    Masseverleihung: Die W- und Z-Eichfelder erhalten durch das Higgs-Feld Masse, was die Reichweite der schwachen Kraft auf extrem kurze Distanzen beschränkt.

    Teilchenerzeugung: Eichfelder können auch Teilchen erzeugen (z.B. ein Photon kann zu einem Elektron-Positron-Paar zerfallen).
    """
    def __init__(
            self,
            g_utils: GUtils,
            qfn_id,
            neighbors_pm,
            attr_keys,
            attrs,
            env,
            time=0.0,
            **args
    ):
        super().__init__()
        self.attrs = self.restore_selfdict(attrs)
        # LOGGER.info("init GaugeBase")
        for k, v in self.attrs.items():
            setattr(self, k, v)
            # LOGGER.info(f"{k}:{v}")

        self.g_utils:GUtils = g_utils
        self.qf_utils = QFUtils(g_utils)

        if getattr(self, "field_key", None) is None:
            setattr(self, "field_key", self._field_value(getattr(self, "type", None)))

        # LOGGER.info("self.field_key", self.field_key)
        self.field_value = getattr(self, self.field_key, None)  # (4,8)

        if self.attrs.get("f_abc") is None:
            f_abc = self.compute_fabc_from_generators()
            setattr(self, "f_abc", f_abc)

        self.calculator = Calculator(
            g_utils
        )

        if self.type.lower() == "gluon":
            self.parent = self.g_utils.get_single_neighbor_nx(
                self.id,
                "GLUON_BUCKET"
            )

        self.symmetry_group_class = SymMain(groups=getattr(self, "_symmetry_groups", []))
        self.qf_utils = QFUtils(self.g_utils)

        # If gauge item: request
        self.neighbor_request_id = self.id if self.type.lower() in self.gauge_fields else self.parent[0]

        # Get neighbors
        self.neighbors = [
            (nid, self.restore_selfdict(nattrs))
            for nid, nattrs in self.g_utils.get_neighbor_list(
                self.id,
                self.qf_utils.all_sub_fields
            )
        ]
        setattr(self, "is_gluon", getattr(self, "type", "").lower() == "gluon")

        self.env=env
        self.neighbors_pm=neighbors_pm
        self.d = env["d"]
        self.attr_keys = attr_keys
        self.time = time
        self.dpsi = None
        self.qf_utils = QFUtils(self.g_utils)
        self.qfn_id = qfn_id

        self.generators = [0.5 * lam for lam in self.su3_group_generators]

        self.vertex = Vertex(
            g_utils,
            self.neighbors,
            parent=self,
            qf_utils=self.qf_utils,
        )



    def main(self):
        # Set prev value
        field_key = getattr(self, "field_key", None)
        nid = getattr(self, "id", None)

        prev_val = getattr(self, f"prev_{field_key}", None)
        g = getattr(self, "g", None)
        ntype = getattr(self, "type", None)
        #print(">>> type", ntype)
        theta_W = self.env["theta_W"]
        d_field_value_key = f"dmu_{field_key}"
        f_abc = getattr(self, "f_abc")
        field_value = getattr(self, field_key)

        new_j_nu = self.init_j_nu(serialize=False)

        if prev_val is None:
            self.attrs[f"prev_{field_key}"] = self.field_value

        snapshot_field_value = self.field_value

        # Main calcs
        j_nu = self._j_nu(nid, ntype, theta_W, g, new_j_nu)
        #self._self_interaction(sym_group)

        # Dmu
        self._dmu_field_value(field_key, d_field_value_key)
        dmu_F_mu_nu = self._dmu_fmunu(field_key)

        # Block 2
        F_mu_nu = self._F_mu_nu(
            d_field_value=getattr(self, d_field_value_key),
            g=g,
            ntype=ntype,
            f_abc=f_abc,
            field_value=getattr(self, field_key),
        )

        # VERTEX
        self.vertex.main()

        # FIELD VALUE
        self._update_gauge_field(
            field_value,
            field_key,
            nid,
            F_mu_nu,
            j_nu,
            f_abc,
            dmu_F_mu_nu,
            ntype,
        )

        # G -> G interaction
        """self._GG_coupling(
            ntype,
            field_value,
            nid,
            theta_W,
            g,
            field_key
        )"""

        """self.triple_coupler._triples(
            field_value, field_key, ntype
        )

        self.quad_coupler._quads(
            field_value, field_key, ntype
        )"""

        # finish
        self.attrs[f"prev_{field_key}"] = snapshot_field_value


        new_dict = check_serialize_dict(
            self.__dict__,
            self.attr_keys
        )

        self.g_utils.update_node(new_dict)
       #print(f"Update for {nid} finished")

    def _check_dict(self):
        for k,v in self.__dict__.items():
           print(f"{k}:{v}")

    def _quads(self):
        pass


    def _dmu_field_value(self, field_key, d_field_value_key):
        dmu = self.calculator.cutils._dmu(
            self_ntype=self.type,
            attrs=self.attrs,
            d=self.d,
            neighbors_pm=self.neighbors_pm,
            field_key=field_key
        )

        #print("var_key", d_field_value_key, "=", dmu)
        setattr(self, d_field_value_key, dmu)
        #print("CHECK attr[var_key]", self.__dict__[d_field_value_key])

    def _dmu_fmunu(self, field_key):
        # dmuF(mu_nu)
        dmu = self.calculator.cutils._dmu(
            self_ntype=self.type,
            attrs=self.attrs,
            d=self.d,
            neighbors_pm=self.neighbors_pm,
            field_key=field_key
        )
        setattr(self, "dmu_F_mu_nu", dmu)
        return dmu



    def _j_nu(self, nid, ntype,theta_W, g, new_j_nu):
        """
        J_nu ersetzt die kopplungsterme auf gauge seite
        für higg u ferms gelten ander regeln
        Berechnet j_nu für alle gekoppelten Materiefelder.
        Jnu gibt an wie viele operationen im system (feld) ausgeführt werden.
        Berücksichtigt automatisch:
          - Leptonen (nur U(1) + SU(2))
          - Quarks (U(1) + SU(2) + SU(3))
        """
        #print("calc jnu for", nid)
        ferm_neighbors = self.g_utils.get_neighbor_list(nid, target_type=FERMIONS)
        # LOGGER.info("# ferm_neighbors", len(ferm_neighbors))
        for fermid, fattrs in ferm_neighbors:

            # GET ATTRS ###############
            fattrs = self.restore_selfdict(fattrs)
            nntype = fattrs.get("type")
            #print(f"{ntype} fermon euighbor: {nntype}")
            psi = fattrs.get("psi")
            charge= fattrs.get("charge")
            psi_bar= fattrs.get("psi_bar")
            g = getattr(self, "g")
            isospin=fattrs.get("isospin")
            # Calc J_nu
            new_j_nu = np.zeros((4,), dtype=complex)
            #print("else nntype:", nntype)


            # KB Single Gloun item needs 3,4 Dirac
            # Quark -> photon (etc) needs 4, Dirac
            gluon_index=None
            if ntype.lower() == "gluon":
                gluon_index = getattr(self, "gluon_index")

            if ntype.lower() != "gluon" and "quark" in nntype.lower():
                psi = psi[0]
                psi_bar=psi_bar[0]
            new_j_nu = self._j_nu_process(
                    new_j_nu,
                    psi=psi,
                    charge=charge,
                    psi_bar=psi_bar,
                    g=g,
                    isospin=isospin,
                    ntype=ntype,
                    gluon_index=gluon_index,
                    nntype=nntype
                )

        if ntype.lower() != "gluon": # gluon higgs !couple
            hid, hattrs = self.g_utils.get_single_neighbor_nx(nid, "PHI")
            hattrs = self.restore_selfdict(data=hattrs)
            if hid is not None:
                new_j_nu += self.j_nu_higgs(
                    new_j_nu,
                    phi=hattrs["phi"],
                    d_phi=hattrs["d_phi"],
                    g=g,
                    ntype=ntype,
                    theta_w=theta_W
                )
        setattr(self, "j_nu", new_j_nu)
        return new_j_nu

    def _F_mu_nu(self, d_field_value, g, ntype, f_abc, field_value):
        """
        Berechnet F_{mu,nu}^a für beliebige Gauge-Felder.
        jedes feld anderes fmunu/gmunu (gluon) etc
        nach ssb nur noch photon
        """
        F_mu_nu = self.f_mu_nu(
            d_field_value, g, ntype, f_abc, field_value
        )
        setattr(self, "F_mu_nu", F_mu_nu)
        return F_mu_nu

    def _update_gauge_field(self, field_value, field_key, nid,  F_mu_nu, j_nu, f_abc, dmu_F_mu_nu, ntype):
        #print("_update_gauge_field", nid)
        new_fv = self.update_gauge_field(
            field_value,
            F_mu_nu,
            j_nu,
            f_abc,
            dmu_F_mu_nu,
            self.d,
            ntype
        )
        setattr(self, field_key, new_fv)
        #print("finished _update_gauge_field")



    def _self_interaction(self, sym_group):
        """
        todo -> include triple and quad g->g couplings
        :param sym_group:
        :return:
        """
        self.symmetry_group_class.sym_classes[sym_group].compute_self_interaction(
            **self.attrs
        )




    def _get_gg_coupling_utils(self, nid) -> dict:
        utils = {}
        intern_neighbors = self.g_utils.get_neighbor_list(nid, trgt_rel="intern_coupling")
        for nnid, nattrs in intern_neighbors:
            ntype = nattrs.get("type").lower()
            nattrs = self.restore_selfdict(nattrs)
            if ntype in ["w_plus", "w_minus"]:
                field_key = self._field_value(type=ntype)
                utils[field_key] = {
                    "field_value": nattrs[field_key],
                    "d": nattrs[f"dmu_{field_key}"]
                }
        return utils











    def tripple_id_list_powerset(self, neighbors):
        id_set = []
        for k,v in neighbors.items():
            id_set.append(list(v.keys()))
        return list(product(id_set[0], id_set[1], id_set[2]))

    def _gluon_interaction(self, field_key):
        """
        gluon kuppeln hardcore untereinander
        ob kupplung besteht
        Wenn auch nur ein Feld =0 ist, wird der Term =0

        Wenn außer abc auch (g,c,b) valid ist –>
        wird die coupling_strength summiert
        :return:
        """
        gluons = []
        # Get gluons
        for nnid, nattrs in self.neighbors:
            ntype = nattrs.get("type").lower()
            if ntype == "gluon":
                gluons.append((nnid, nattrs))

        # Extract all neighbor indices
        all_neighbor_indices = {}
        for nnid, g in gluons:
            gluon_field_value = deserialize_complex(g[field_key])
            # Check active
            all_neighbor_indices[nnid] = self._check_active_indizes(gluon_field_value)








"""


            if nntype.lower() in self.quarks:
                new_j_nu = np.zeros((3, 4,), dtype=complex)
                #print("_j_nu gauge nntype:", nntype)
                for i in range(3):  # auch nach ssb imernoch 3 spinoren
                    # j_nu wird für alle quarks aufsummiert
                    new_j_nu[i] = self._j_nu_process(
                        new_j_nu[i],
                        psi=psi[i],
                        charge=charge,
                        psi_bar=psi_bar[i],
                        g=g,
                        isospin=isospin,
                        ntype=ntype,
                    )
            else:

"""

"""def _GG_coupling(
            self,
            ntype,
            field_value,
            nid,
            theta_w,
            g,
            field_key,
    ):
        j_nu = getattr(self, "j_nu")

        utils = self._get_gg_coupling_utils(nid)

        if ntype.lower() == "photon":
            j_nu_w_photon: Hauptanteil, der immer relevant ist.
            j_nu_ww_aa: zusätzlicher Beitrag bei sehr 
            hohen Feldstärken oder bei Prozessen mit 
            zwei Photonen.
            Komplette Nichtabelsche Feldtheorie:
            ➤ Beide Terme summieren.
            e = self._e(g, theta_w)
            j_nu += self.j_nu_ww_aa(
                W_plus=utils["w_plus"]["field_value"],
                W_minus=utils["w_minus"]["field_value"],
                A_field=field_value,
                e=e
            )
            j_nu += self.j_nu_w_photon(
                W_plus=utils["w_plus"]["field_value"],
                dW_plus=utils["w_plus"]["d"],
                W_minus=utils["w_minus"]["field_value"],
                dW_minus=utils["w_plus"]["d"],
                e=e
            )

        elif ntype.lower() == "z_boson":
            j_nu += self.j_nu_w_z(
                w_plus=utils["w_plus"]["field_value"],
                dw_plus=utils["w_plus"]["d"],
                w_minus=utils["w_minus"]["field_value"],
                dw_minus=utils["w_minus"]["d"],
                g=g,
                theta_w=theta_w
            )

        elif ntype.lower() == "gluon":
            self._gluon_interaction(field_key)

        setattr(self, "j_nu", j_nu)"""






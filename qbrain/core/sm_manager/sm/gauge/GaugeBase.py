from jax import jit

from _admin._ray_core.utils.ray_validator import RayValidator

from itertools import product

import jax.numpy as jnp

from qbrain.core.sm_manager.sm.gauge.gauge_utils import GaugeUtils

"""
#Vertex
TripleCoupler,
QuadCoupler,
"""
class GaugeBase(
    GaugeUtils,
    RayValidator,

):
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
    photon_field_value = jnp.array([
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
            env,
            g_utils=None
    ):
        self.g_utils=g_utils
        RayValidator.__init__(self, host=None, g_utils=g_utils)
        self.env=env

    @jit
    def main(
            self,
            attrs:dict,
            neighbor_pm_val_same_type,
            neighbor_pm_val_fmunu,
            f_neighbors:dict[str, dict],
            g_neighbors:dict[str, dict],
    ):
        # ferm->gauge coupling
        self.gf_coupling(attrs, f_neighbors)
        # VERTEX (gauge -> gauge coupling)
        self.gg_coupling(attrs, g_neighbors)

        # Dmu
        self._dmu_field_value(
            attrs,
            neighbor_pm_val_same_type
        )

        # Fmunu
        self._F_mu_nu(attrs)

        #dmu fmunu
        self._dmu_fmunu(
            attrs,
            neighbor_pm_val_fmunu
        )

        # FIELD VALUE
        self.update_gauge_field(
            attrs
        )
        return attrs

    def gg_coupling(self, attrs, g_neighbors):
        g_self = attrs.get("g", 1.0)
        field_key = attrs["field_key"]
        field_v_self = attrs[field_key]
        new_j_nu = 0.0
        for _, ndata in g_neighbors.items():
            field_key_n = ndata["field_key"]
            field_v_n = jnp.array(attrs[field_key_n])
            g_n = ndata.get("g")

            # symmetric effective coupling strength
            g_eff = 0.5 * (g_self + g_n)

            # simplified antisymmetric field interaction term: [Aμ, Aν]
            j_pair = g_eff * (field_v_self * field_v_n - field_v_n * field_v_self)

            j_nu = attrs["j_nu"] + jnp.sum(j_pair) * field_v_self / (1e-9 + jnp.linalg.norm(field_v_self))

            new_j_nu+=j_nu

        attrs["j_nu"] = new_j_nu



    def _dmu_fmunu(
            self,
            attrs,
            neighbor_pm_val_fmunu
    ):
        # dmuF(mu_nu)
        field_key = "F_mu_nu"
        dmufmunu = self.call(
            method_name="_dX",
            attrs=attrs,
            d=self.env["d"],
            neighbor_pm_val_same_type=neighbor_pm_val_fmunu,
            field_key=field_key
        )
        attrs[f"dmu_{field_key}"] = dmufmunu


    def _dmu_field_value(
            self,
            attrs,
            neighbor_pm_val_same_type
    ):
        field_key= attrs["field_key"]
        d_field_value_key = attrs[f"dmu_{field_key}"]

        dmu = self.call(
            method_name="_dX",
            attrs=attrs,
            d=self.env["d"],
            neighbor_pm_val_same_type=neighbor_pm_val_same_type,
            field_key=field_key
        )
        attrs[d_field_value_key] = dmu


    def gf_coupling(self, attrs, f_neighbors):
        """
        J_nu ersetzt die kopplungsterme auf gauge seite
        für higg u ferms gelten ander regeln
        Berechnet j_nu für alle gekoppelten Materiefelder.
        Jnu gibt an wie viele operationen im system (feld) ausgeführt werden.
        Berücksichtigt automatisch:
          - Leptonen (nur U(1) + SU(2))
          - Quarks (U(1) + SU(2) + SU(3))
        """

        new_j_nu = jnp.zeros((4,), dtype=complex)

        g = attrs["g"]
        ntype = attrs["type"]

        print("Calc j_nu")
        for nnid, fattrs in f_neighbors.items():
            # ATTRS
            fattrs = self.restore_selfdict(fattrs)
            nntype = fattrs.get("type")
            psi = fattrs.get("psi")
            charge= fattrs.get("charge")
            psi_bar= fattrs.get("psi_bar")
            isospin=fattrs.get("isospin")

            # Calc J_nu
            new_j_nu = jnp.zeros((4,), dtype=complex)

            gluon_index=None
            if ntype.lower() == "gluon":
                gluon_index = getattr(self, "gluon_index")

            if ntype.lower() != "gluon" and "quark" in nntype.lower():
                psi = psi[0]
                psi_bar = psi_bar[0]

            new_j_nu += self._j_nu_process(
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
        attrs["j_nu"] = new_j_nu



    def _F_mu_nu(self, attrs):
        """
        Berechnet F_{mu,nu}^a für beliebige Gauge-Felder.
        jedes feld anderes fmunu/gmunu (gluon) etc
        nach ssb nur noch photon
        """
        field_key = attrs["field_key"]
        d_field_value = attrs[f"dmu_{field_key}"]
        g = attrs["g"]
        ntype = attrs["type"]
        field_value = attrs[field_key]
        F_mu_nu = self.f_mu_nu(
            d_field_value,
            g,
            ntype,
            field_value
        )
        attrs["F_mu_nu"] = F_mu_nu


    @jit
    def update_gauge_field(
            self,
            attrs,
    ):
        """
        ∂_μ F^{μν} = j^ν
        """
        field_key = attrs["field_key"]
        dmu_F_mu_nu = attrs["dmu_F_mu_nu"]
        j_nu = attrs["j_nu"]
        field_value = attrs[field_key]

        field_v_new = field_value + self.env["d"]["t"] * (dmu_F_mu_nu - j_nu)
        attrs[field_key] = field_v_new

    def tripple_id_list_powerset(self, neighbors):
        id_set = []
        for k,v in neighbors.items():
            id_set.append(list(v.keys()))
        return list(product(id_set[0], id_set[1], id_set[2]))




"""

def _gluon_interaction(self, field_key):
    gluon kuppeln hardcore untereinander
    ob kupplung besteht
    Wenn auch nur ein Feld =0 ist, wird der Term =0

    Wenn außer abc auch (g,c,b) valid ist –>
    wird die coupling_strength summiert
    :return:
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



        if prev_val is None:
            self.attrs[f"prev_{field_key}"] = self.field_value
            
        # Set prev fmunu
        prev_fmunu = "F_mu_nu_prev"
        if getattr(self, prev_fmunu, None) is None:
            attrs[prev_fmunu] = getattr(self, "F_mu_nu")
        # HIGGS
        if ntype.lower() not in ["gluon", "photon"]:  # gluon higgs !couple
            hid, hattrs = self.call(
                method_name="get_single_neighbor_nx",
                node=nid,
                target_type="PHI",
            )
            hattrs = self.restore_selfdict(admin_data=hattrs)

            if hid is not None:
                new_j_nu += self.j_nu_higgs(
                    new_j_nu,
                    phi=hattrs["phi"],
                    d_phi=hattrs["d_phi"],
                    g=g,
                    ntype=ntype,
                    theta_w=self.env["theta_W"]
                )
        attrs["j_nu"] = new_j_nu
        print(f"jν: {new_j_nu}")
"""
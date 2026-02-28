from _admin._ray_core.utils.ray_validator import RayValidator
from qbrain.core.sm.gauge.gauge_utils import GaugeUtils
from qbrain.utils.serialize_complex import deserialize_complex, check_serialize_dict
from itertools import product



class GaugeBase(
    GaugeUtils,
):
    def __init__(self):
        GaugeUtils.__init__(self)

    def main(
            self,
            attrs: dict,
            neighbor_pm_val_same_type,
            neighbor_pm_val_fmunu,
            f_neighbors: dict[str, dict],
            g_neighbors: dict[str, dict],
    ):
        self.coupling(
            attrs,
            f_neighbors,
            g_neighbors,
        )

        # Dmu
        self._dmu_field_value(
            attrs,
            neighbor_pm_val_same_type
        )

        # Fmunu
        self._F_mu_nu(attrs)

        # dmu fmunu
        self._dmu_fmunu(
            attrs,
            neighbor_pm_val_fmunu
        )

        # FIELD VALUE
        self.update_gauge_field(
            attrs
        )
        return attrs









    def tripple_id_list_powerset(self, neighbors):
        id_set = []
        for k, v in neighbors.items():
            id_set.append(list(v.keys()))
        return list(product(id_set[0], id_set[1], id_set[2]))




class TGaugeBase(
    GaugeUtils,
    RayValidator
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
            ntype,
            env,
            g_utils=None
    ):
        self.g_utils=g_utils
        RayValidator.__init__(self, host=None, g_utils=g_utils)
        GaugeUtils.__init__(self)
        self.field_symbol = self.get_gauge_field_symbol(ntype)

        self.d = env["d"]
        self.dpsi = None

        self.vertex = Vertex(
            parent=self,
        )
        print("GaugeBase initialized")

    def main(
            self,
            env,
            attrs,
            all_subs,
            neighbor_pm_val_same_type,
            neighbor_pm_val_fmunu,
    ):
        # args
        self.env=env
        self.all_subs = all_subs
        self.neighbor_pm_val_same_type = neighbor_pm_val_same_type
        self.neighbor_pm_val_fmunu=neighbor_pm_val_fmunu

        # attrs
        self.attrs = self.restore_selfdict(attrs)

        # field key & value
        ntype = getattr(self, "type", None)
        field_key = getattr(self, "field_key", None)
        if field_key is None:
            field_key = self._field_value(ntype)
            setattr(self, "field_key", field_key)
        self.field_value = self.attrs.get(getattr(self, "field_key", None))

        # Set prev field
        prev_key = f"{field_key}_prev"
        if getattr(self, prev_key, None) is None:
            self.attrs[prev_key] = self.field_value

        # Set prev fmunu
        prev_fmunu = "F_mu_nu_prev"
        if getattr(self, prev_fmunu, None) is None:
            self.attrs[prev_fmunu] = getattr(self, "F_mu_nu")

        # fabc
        if self.attrs.get("f_abc") is None:
            f_abc = self.compute_fabc_from_generators()
            setattr(self, "f_abc", f_abc)
        else:
            f_abc = getattr(self, "f_abc")

        # Set prev value
        field_key = getattr(self, "field_key", None)
        nid = getattr(self, "id", None)

        prev_val = getattr(self, f"prev_{field_key}", None)
        g = getattr(self, "g", None)
        theta_W = self.env["theta_W"]
        d_field_value_key = f"dmu_{field_key}"
        field_value = getattr(self, field_key)

        new_j_nu = self.init_j_nu(serialize=False)

        if prev_val is None:
            self.attrs[f"prev_{field_key}"] = self.field_value

        snapshot_field_value = self.field_value

        # Main calcs
        j_nu = self._j_nu(nid, ntype, theta_W, g, new_j_nu)

        # Dmu
        self._dmu_field_value(field_key, d_field_value_key)
        dmu_F_mu_nu = self._dmu_fmunu(field_key="F_mu_nu")

        # Block 2
        F_mu_nu = self._F_mu_nu(
            d_field_value=getattr(self, d_field_value_key),
            g=g,
            ntype=ntype,
            f_abc=f_abc,
            field_value=getattr(self, field_key),
        )

        # VERTEX
        self.vertex.main(
            neighbors=self.all_subs["GAUGE"],
        )

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
        # finish
        self.attrs[f"prev_{field_key}"] = snapshot_field_value

        new_dict = check_serialize_dict(
            self.attrs,
        )

        return new_dict
        #print(f"Update for {nid} finished")


    def _dmu_field_value(self, field_key, d_field_value_key):
        dmu = self.call(
            method_name="_dX",
            attrs=self.attrs,
            d=self.d,
            neighbor_pm_val_same_type=self.neighbor_pm_val_same_type,
            field_key=field_key
        )

        #print("var_key", d_field_value_key, "=", dmu)
        setattr(self, d_field_value_key, dmu)
        #print("CHECK attr[var_key]", self.__dict__[d_field_value_key])

    def _dmu_fmunu(self, field_key):
        # dmuF(mu_nu)
        dmu = self.call(
            method_name="_dX",
            attrs=self.attrs,
            d=self.d,
            neighbor_pm_val_same_type=self.neighbor_pm_val_fmunu,
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

        #print("Calc j_nu")
        for nntype, neighbor_struct in self.all_subs["FERMION"].items():
            for nnid, fattrs in neighbor_struct.items():
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
        # HIGGS
        #print(">>>ntype", ntype)
        #print(">>>self.g", self.g_utils)
        #print(">>>self.host", self.host)
        if ntype.lower() not in ["gluon", "photon"]:  # gluon higgs !couple
            hid, hattrs = self.call(
                method_name="get_single_neighbor_nx",
                node=nid,
                target_type="PHI",
            )
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
        print(f"jν: {new_j_nu}")
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

        print(f"{self.field_symbol}: {new_fv}")
        setattr(self, field_key, new_fv)





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



import jax

from data import FERM_PARAMS
from sm.fermion.ferm_utils import FermUtils

import jax.numpy as jnp

class FermCreator(FermUtils):

    def __init__(self, g):
        super().__init__()
        self.g=g
        self.layer = "PSI"
        self.parent=["FERMION", self.layer]

    def create_ferm_attrs(
            self,
            ntype,
            px_id,
            pos,
            light=False,
            nid=None,
    ) -> list:
        attrs_struct = []
        try:
            if "quark" in ntype.lower():
                for item_index in range(3):
                    if nid is None:
                        nid = f"{ntype}__{px_id}__{item_index}"
                    else:
                        item_index = int(nid.split("_")[-1])

                    attrs_struct.append(
                        self.get_attrs_core(
                            px_id,
                            nid,
                            ntype,
                            pos,
                            item_index,
                        )
                    )
                    nid=None
            else:
                fermid = f"{ntype}__{px_id}"
                attrs_struct.append(
                    self.get_attrs_core(
                        px_id,
                        fermid,
                        ntype,
                        pos,
                    )
                )
            nid = None
        except Exception as e:
            print(f"Err create_ferm_attrs: {e}")
        return attrs_struct


    def get_attrs_core(
            self,
            px_id,
            nid,
            ntype,
    ):
        """

        nid=nid,
        tid=0,
        gterm=0,
        yterm=0,
        parent=self.parent,
        type=ntype,

        """
        # todo parallalize array creation for all F and px
        attr_struct = dict(
            nid=nid,
            tid=0,
            type=ntype,
            parent=self.parent,
            px=px_id,
        )
        return attr_struct

    def create_f_core(self, pos, item, just_v=False, just_k=False):
        psi = self.field_value(pos)
        field = dict(
            gterm=0,
            yterm=0,
            gf_coupling=psi,
            gg_coupling=psi,
            dmu_psi=self.dmu(pos),
            psi=psi,
            dirac=psi,
            psi_bar=psi,
            prev=psi,
            #quark_index=item_index,
            velocity=0.0,
            **item,
        )
        if just_v:
            return field.values()
        if just_k:
            return field.keys()
        return jax.device_put(field)

    def create_f_core_batch(
            self,
            ntype,
            amount_nodes,
            dim,
            just_v=False,
            just_k=False,
    ):
        psi = self.field_values(amount_nodes, dim)
        item = FERM_PARAMS[ntype]
        field = dict(
            gterm=self.field_value(dim),
            yterm=self.field_value(dim),

            gf_coupling=psi,
            gg_coupling=psi,

            dmu_psi=jnp.stack([self.dmu(amount_nodes, dim) for _ in range(len(amount_nodes))]),

            psi=psi,
            dirac=psi,
            psi_bar=psi,
            prev=psi,

            velocity=self.field_value(dim),

            **item,
        )

        if just_v:
            return field.values()
        if just_k:
            return field.keys()
        return jax.device_put(field)

    def build_


    def create_quark(self, pos, item, just_v=False):
        psi = self.quark_field(pos)
        field = dict(
            gterm=0,
            yterm=0,
            gf_coupling=psi,
            gg_coupling=psi,
            dmu_psi=self.dmu(),
            psi=psi,
            dirac=psi,
            psi_bar=psi,
            prev=psi,
            #quark_index=item_index,
            velocity=0.0,
            **item,
        )
        if just_v:
            return field.values()
        return field



    def create(self, src_qfn_id):
        # PSI
        for ferm_field, attrs in FERM_PARAMS.items():
            print(f"Create {ferm_field} for {src_qfn_id}")
            ferm_field = ferm_field.upper()
            self._create_quark_parent(
                ferm_field,
                src_qfn_id,
                attrs
            )
        self.connect_quark_doublets(src_qfn_id)
        print("Fermions created and Quarks connected")

    def connect_quark_doublets(self, src_qfn_id):
        """
        # die drei quark-paare (up+down, charm+strange, top+bottom) sind eigenständige felder
        # → jedes paar bildet ein separates SU(2)_L-dublett (auch nach SSB)
        # → jedes dublett hat eigene komponenten und eigene masse

        #  immer nur EIN quark-doublet koppelt direkt an ein W⁺ oder W⁻ vertex (pro punkt im raum)
            aber es kann sein dass bottom up an w+ und cham strange an w- zur selben zeit koppelt!

        # W±-kopplung überlagert sie jedoch durch die CKM-matrix:
        # → z. B. W⁺ koppelt u → d, s, b (nicht nur d)
        # → realisiert über linearkombination: d' = V_ud * d + V_us * s + V_ub * b

        # welches dublett an ein W⁺ koppelt, wird durch die QUANTENZAHLEN UND DIE DYNAMIK des prozesses bestimmt

        #<<< kupplungs faktoren (was legt fest welches dubet an w+/- kuppelt?: >>>

        # 1. verfügbare teilchen im prozess
        #    → wenn z. B. ein top-quark erzeugt wurde, kann (t, b) koppeln

        # 2. energie des systems
        #    → schwere dubletts (z. B. (t, b)) benötigen mehr energie
        #    → bei niedriger energie sind nur (u, d), (c, s) relevant

        # 3. CKM-Matrix
        #    → legt fest, wie stark ein up-typ quark mit jedem down-typ koppelt
        #    → z. B. u → d,s,b mit gewichtung (V_ud, V_us, V_ub)

        # 4. erhaltungsgrößen
        #    → ladung, farbe, energie, impuls etc. müssen im vertex erhalten bleiben

        # zusammengefasst:
        # das universelle W⁺-Feld kann mit allen dubletts koppeln
        # welches tatsächlich koppelt, hängt von teilchenzustand und erlaubten wechselwirkungen ab
        """
        partner_map = {
            "up": "down",
            "down": "up",
            "charm": "strange",
            "strange": "charm",
            "top": "bottom",
            "bottom": "top",
        }


        # Get Partners and connect
        for p1, p2 in partner_map.items():
            src_layer = f"{p1}_quark".upper()
            trgt_layer = f"{p2}_quark".upper()

            src_id, src_attrs = self.g.get_single_neighbor_nx(src_qfn_id, src_layer)
            trgt_id, trgt_attrs = self.g.get_single_neighbor_nx(src_qfn_id, trgt_layer)

            # P1 -> P2
            self.g.add_edge(
                src_id,
                trgt_id,
                attrs=dict(
                    rel=f"doublet_partner",
                    src_layer=src_layer,
                    trgt_layer=trgt_layer,
                )
            )


    def _create_quark_parent(
            self,
            ferm_field,
            src_qfn_id,
            attrs,
    ):
        print(f"Create parent FermField {ferm_field}")
        fermid = f"{ferm_field}_{src_qfn_id}"

        parent_attrs = None # todo

        self.g.add_node(attrs=parent_attrs)

       # PSI -> PIXEL
        self.g.add_edge(
            src_qfn_id,
            f"{fermid}",
            attrs=dict(
                rel=f"has_field",
                src_layer="PIXEL",
                trgt_layer=ferm_field,
            )
        )
        print(f"created {fermid}")
        return fermid



    def psi_x_bar(self, ferm_field):
        #  for general ferms and single quark item
        psi = self._init_psi(ntype=ferm_field)
        psi_bar = self._init_psi(ntype=ferm_field)
        return psi, psi_bar


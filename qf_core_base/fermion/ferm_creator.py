from qf_sim.physics.quantum_fields.nodes.fermion import FERM_PARAMS, PSI_UNIFORM
from qf_sim.physics.quantum_fields.nodes.fermion.ferm_utils import FermUtils
from utils._np.serialize_complex import serialize_complex


class FermCreator(FermUtils):


    def __init__(self, g):
        super().__init__()
        self.g=g
        self.layer = "PSI"
        self.parent=[self.layer, "FERMION"]


    def create(self, src_qfn_id):
        # PSI
        for ferm_field, attrs in FERM_PARAMS.items():

            ferm_field = ferm_field.upper()
            self._create_quark_parent(
                ferm_field,
                src_qfn_id,
                attrs
            )
            """if "quark" in ferm_field.lower():
                self._create_quark_items(
                    ferm_field,
                    attrs,
                    src_qfn_id,
                )

            else:
                self._create_quark_parent(
                    ferm_field,
                    src_qfn_id,
                    attrs
                )"""

        print("Psi created")


    def _create_quark_items(
        self,
        f_field,
        fattrs,
        src_qfn_id,
    ):
        symmetry_groups = self.get_sym_group(f_field)

        for i in range(3):
            # ID tyüe
            nid = f"{f_field}_{i}_{src_qfn_id}"
            f_field = f_field.upper()

            # Field values
            psi, psi_bar = self.psi_x_bar(f_field)

            attrs = dict(
                id=nid,
                parent=self.parent + ["QUARK"],
                type=f_field,
                _symmetry_groups=symmetry_groups,
                time=0.0,
                sub_type="ITEM",
                psi=serialize_complex(com=psi),
                psi_bar=serialize_complex(com=psi_bar),
                **PSI_UNIFORM,
                **fattrs,
            )

            self.g.add_node(attrs=attrs)

            # Parent -> Child Edge
            self.g.add_edge(
                src_qfn_id,
                nid,
                attrs=dict(
                    rel="has_instance",
                    src_layer="QFN",
                    trgt_layer=f_field,
                )
            )

    def _create_quark_parent(
            self,
            ferm_field,
            src_qfn_id,
            attrs,
    ):
        print("Create parent Ferm - Field")
        fermid = f"{ferm_field}_{src_qfn_id}"

        psi, psi_bar = self.psi_x_bar(ferm_field)
        parent_attrs = dict(
                id=fermid,
                type=ferm_field,
                parent=self.parent,
                time=0.0,
                psi=serialize_complex(com=psi),
                psi_bar=serialize_complex(com=psi_bar),
                _symmetry_groups=self.get_sym_group(ferm_field),
                **PSI_UNIFORM,
                **attrs,
            )

        self.g.add_node(attrs=parent_attrs)

       # PSI -> QFN
        self.g.add_edge(
            src_qfn_id,
            f"{fermid}",
            attrs=dict(
                rel=f"has_field",
                src_layer="QFN",
                trgt_layer=ferm_field,
            )
        )
        return fermid



    def psi_x_bar(self, ferm_field):
        #  for general ferms and single quark item
        psi = self.init_psi(ntype=ferm_field, serialize=False)
        psi_bar = self._psi_bar(psi, self._is_quark(ferm_field))
        return psi, psi_bar


"""
        if "quark" in ferm_field.lower():
            parent_attrs = dict(
                id=f"{fermid}",
                parent=self.parent,
                type=f"{ferm_field}",
                sub_type="BUCKET",
                time=0.0,
            )
            


"""
import numpy as np

from data import GAUGE_FIELDS
from qbrain.core.sm.gauge.gauge_utils import GaugeUtils

class GaugeCreator(GaugeUtils):
    def __init__(self, g_utils):
        """
        gu: Utility-Objekt mit init_G, init_fmunu, etc.
        graph: Dein Graph-Objekt mit add_node, add_edge
        layer: Aktueller Layer-Name
        gauge_fields: Dict der Gauge-Felder (GAUGE_FIELDS)
        """
        super().__init__()
        self.g = g_utils
        self.qfn_layer = "PIXEL"
        self.gluon_item_type = "GLUON"

    def get_gauge_params(
            self,
            ntype,
            pos,
            px_id: str,
            light=None,
            id=None,
    ) -> list:

        field_key = self._field_value(ntype)

        try:
            attrs_struct = []
            if "gluon" == ntype.lower():
                for item_index in range(8):
                    if nid is None:
                        nid = f"{ntype}__{px_id}__{item_index}"
                    else:
                        item_index = int(nid.split("_")[-1])

                    attrs = self.get_g_params_core(
                        pos,
                        nid,
                        ntype,
                        field_key,
                        item_index,
                        light
                    )

                    self.check_extend_attrs(
                        ntype,
                        attrs
                    )

                    attrs_struct.append(attrs)
                    nid = None
            else:
                if nid is None:
                    nid = f"{ntype}__{px_id}"

                attrs = self.get_g_params_core(
                    pos,
                    nid,
                    ntype,
                    field_key,
                    item_index=None
                )

                self.check_extend_attrs(
                    ntype,
                    attrs,
                )

                attrs_struct.append(attrs)
            return attrs_struct
        except Exception as e:
            print(f"Err get_gauge_params: {e}")


    def check_extend_attrs(
            self,
            ntype,
            attrs,
    ):
        if ntype.lower() == "gluon":
            attrs["parent"].append(ntype.upper())

    def get_g_params_core(
            self,
            pos,
            nid,
            ntype,
            field_key,
            item_index,
    ):
        attrs = dict(
            id=nid,
            tid=0,
            parent=["GAUGE"],
            type=ntype,
            field_key=field_key,
            **self.gfield(pos, item_index),
        )
        return attrs

    def gfield(
            self,
            ntype,
            dim
    ):
        field_value=self.field_value(dim=dim)

        const = {
            k: v
            for k, v in GAUGE_FIELDS[ntype].items()
        }

        field = {
            "gg_coupling": field_value,
            "gf_coupling": field_value,
            "field_value": field_value,
            "prev_field_value": field_value,

            "j_nu": field_value,

            "dmuG": self.dmu(dim),
            "fmunu": self.fmunu(dim),
            "prev_fmunu": self.fmunu(dim),
            "dmu_fmunu": self.dmu_fmunu(dim),
            **const,
        }

        return field


    def get_gluon(self, pos):
        field_value=self.gluon_fieldv(pos)

        return dict(
            gg_coupling=0,
            gf_coupling=0,
            field_value=field_value,
            prev=0,
            j_nu=0,
            dmuG=self.dmu(pos),
            fmunu=self.fmunu(pos),
            prev_fmunu=self.fmunu(pos),
            dmu_fmunu=self.dmu_fmunu(pos),
            charge=0,
            mass=0.0,
            g=1.217,
            spin=1,
        )


    def create(self, src_qfn_id):
        for g_field, gattrs in GAUGE_FIELDS.items():
            g_field = g_field.upper()

            if g_field.lower() == "gluon":
                self._create_gluon_items(
                    g_field,
                    gattrs,
                    src_qfn_id,
                )
            else:
                self._create_gauge(
                    g_field,
                    src_qfn_id,
                    gattrs
                )

            print(f"{g_field} for {src_qfn_id} created")


    def connect_intern_fields(self, pixel_id):
        for src_field, trgt_fields in self.gauge_to_gauge_couplings.items():
            if src_field.lower() != "gluon": # -> alread connected in gluon process
                src_id, src_attrs = self.g.get_single_neighbor_nx(pixel_id, src_field.upper())

                for trgt_field in trgt_fields:
                    trgt_id, trgt_attrs = self.g.get_single_neighbor_nx(
                        pixel_id,
                        trgt_field.upper()
                    )
                    self.g.add_edge(
                        src=src_id,
                        trgt=trgt_id,
                        attrs=dict(
                            rel="intern_coupled",
                            src_layer=src_field.upper(),
                            trgt_layer=trgt_field.upper(),
                        )
                    )
        print("local gauges connected")



    def connect_gluons(self, nid):
        """
        Get all neighbors of a single gluon ->
        get their GLUON_ITEMs ->
        connect them all to nid
        """
        # Get all Gluon neighbors
        g_item = "GLUON"
        all_gluon_neighbor_items = self.get_gluon_neighbor_items(
            nid,
            trgt_type=g_item
        )

        # connect nid to all of them
        for ngluon_sub_id, snattrs in all_gluon_neighbor_items:
            self.g.add_edge(
                nid,
                ngluon_sub_id,
                attrs=dict(
                    rel="uses_param",
                    src_layer=g_item,
                    trgt_layer=g_item,
                )
            )



    def get_gluon_neighbor_items(self, nid, trgt_type):
        """
        Receive all possible gluon item cons for
        a single item
        intern & extern
        """
        all_gluon_neighbor_items = []
        # EXTERN
        gluon_neighbors = self.g.get_neighbor_list(
            nid,
            "GLUON"
        )

        for ngluon_id, _ in gluon_neighbors:
            # Get neighbors sub-gluon-fields
            gluon_neighbors_subs = self.g.get_neighbor_list(
                ngluon_id,
                trgt_type
            )
            all_gluon_neighbor_items.extend(gluon_neighbors_subs)

        return all_gluon_neighbor_items




    def _create_gluon_items(
        self,
        g_field,
        gattrs,
        src_qfn_id,
        pos,
        ntype,
    ):

        all_gluon_ids = set()
        for i in range(8):
            gauge_id = f"{g_field}_{i}_{src_qfn_id}"
            all_gluon_ids.add(gauge_id)

            attrs = self.get_gauge_params(
                pos=pos,
                id=gauge_id,
                px_id=gattrs[src_qfn_id],
                ntype=ntype
            )
            self.g.add_node(
                attrs=attrs
            )


        # Connect intern each gluon
        for gluon_id in all_gluon_ids:
            for trgt_gluon_id in all_gluon_ids:
                if gluon_id != trgt_gluon_id:
                    self.g.add_edge(
                        src=gluon_id,
                        trgt=trgt_gluon_id,
                        attrs=dict(
                            rel="intern_gluon",
                            src_layer=self.gluon_item_type,
                            trgt_layer=self.gluon_item_type,
                        )
                    )
                    print(f"Connect {gluon_id} -> {trgt_gluon_id}, {self.g.G.has_edge(gluon_id, trgt_gluon_id)}")

        self.g.print_edges("GLUON", "GLUON")
        print("Gluon items created")

    def _create_gauge(
        self,
        g_field,
        src_qfn_id,
        attrs
    ):

        gauge_id = f"{g_field}_{src_qfn_id}"

        parent_attrs = self.get_gauge_params(
            id=gauge_id,
            ntype=g_field,
            gluon_index=None,
            gattrs=attrs,
        )

        self.g.add_node(attrs=parent_attrs)

        return parent_attrs


    def _connect_2_qfn(self, src_qfn_id, gauge_id, g_field):
        # PIXEL -> Parent Edge
        self.g.add_edge(
            src_qfn_id,
            gauge_id,
            attrs=dict(
                rel="has_field",
                src_layer=self.qfn_layer,
                trgt_layer=g_field,
            )
        )

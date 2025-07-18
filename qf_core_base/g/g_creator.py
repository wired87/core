from qf_core_base.g import GAUGE_FIELDS
from qf_core_base.g.gauge_utils import GaugeUtils


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
        self.qfn_layer = "QFN"
        self.gluon_item_type = "GLUON"

    def create(self, src_qfn_id):
        g = "GAUGE"
        gluon = "GLUON"

        for g_field, gattrs in GAUGE_FIELDS.items():
            g_field = g_field.upper()
            # Parent Node erstellen

            # Falls Gluon, Child Nodes erzeugen
            if g_field.lower() == "gluon":
                self._create_gluon_items(
                    g_field,
                    gattrs,
                    src_qfn_id,
                    g,
                    gluon,
                )
            else:
                self._create_g_parent(
                    g_field,
                    src_qfn_id,
                    g,
                    gattrs
                )

            print(f"{g_field} for {src_qfn_id} created")


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
        # INTERN
        intern_gloun_items = self.g.get_neighbor_list(
            nid,
            trgt_type
        )
        for intern_g_item_id, gattrs in intern_gloun_items:
            if nid != intern_g_item_id:
                all_gluon_neighbor_items.append(
                    (intern_g_item_id, gattrs)
                )
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



    def _get_gauge_base_payload(self, g_field):
        field_value = self.init_G(ntype=g_field)
        field_key = self._field_value(g_field)
        dmu_field_key = self.init_fmunu(ntype=g_field)
        fmunu = self.init_fmunu(ntype=g_field)
        j_nu = self.init_j_nu()
        #printer(locals())

        return field_key, field_value, dmu_field_key, fmunu, j_nu

    def _create_gluon_items(
        self,
        g_field,
        gattrs,
        src_qfn_id,
        g,
        gluon,
    ):
        for i in range(8):
            gauge_id = f"{g_field}_{i}_{src_qfn_id}"
            g_item_field = g_field.upper()

            field_key, field_value, dmu_field_key, fmunu, j_nu = self._get_gauge_base_payload(
                g_item_field,
            )

            attrs = dict(
                id=gauge_id,
                parent=[g, gluon],
                time=0.0,
                gluon_index=i,
                type=g_item_field,
                sub_type="ITEM",
                j_nu=j_nu,
                F_mu_nu=fmunu,
                **gattrs,
                #**FIELD_METADATA,
            )

            attrs[field_key] = field_value
            attrs[f"dmu_{field_key}"] = dmu_field_key

            self.g.add_node(attrs=attrs)

            # Parent -> Child Edge
            self._connect_2_qfn(src_qfn_id, gauge_id, g_field)

        print("Gluon items created")

    def _create_g_parent(
        self,
        g_field,
        src_qfn_id,
        g,
        attrs
    ):
       #print("Create parent G field")
        symmetry_groups = self.get_sym_group(g_field)

        gauge_id = f"{g_field}_{src_qfn_id}"

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
            _symmetry_groups=symmetry_groups,
            #f_abc=None,
            **attrs,
            #**FIELD_METADATA,
        )
        # Zusätzliche Felder nur für Nicht-Gluon
        parent_attrs[field_key] = field_value
        parent_attrs[f"dmu_{field_key}"] = dmu_field_key

        self.g.add_node(attrs=parent_attrs)

        self._connect_2_qfn(src_qfn_id, gauge_id, g_field)

        return parent_attrs
       # print("All Gluons connected")


    def _connect_2_qfn(self, src_qfn_id, gauge_id, g_field):
        # QFN -> Parent Edge
        self.g.add_edge(
            src_qfn_id,
            gauge_id,
            attrs=dict(
                rel="has_field",
                src_layer=self.qfn_layer,
                trgt_layer=g_field,
            )
        )



"""    def get_gauge_neighbor(self, nid):
   Get all non gluon neighbors for non-gluon gauge
   all_neighbor_items = []
   target_type = [
       *self.g_electroweak,
       *self.g_qed
   ]
   # Get intern G-Fields
   intern_items = self.g.get_neighbor_list(
       nid,
       target_type=target_type
   )
   # Valid intern coupliings
   for innid, inattrs in intern_items:
       if innid != nid:
           all_neighbor_items.append(
               (innid, inattrs)   
           )
        if g_field.lower() == "gluon":
            parent_attrs = dict(
                id=gauge_id,
                parent=[g],
                type=f"{g_field}",
                sub_type="BUCKET",
                _symmetry_groups=symmetry_groups,
                time=0.0,
            )
        else:
   # Valid extern couplings"""
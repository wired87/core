from core.sm.higgs.phi_utils import HiggsUtils

class HiggsCreator(HiggsUtils):
    """
    A minimalistic class for creating a simplified Higgs-like field node
    and connecting it within a graph structure.
    """

    def __init__(self, g):
        HiggsUtils.__init__(self)
        self.g=g

    def higgs_attrs(self, px_id, nid=None) -> list[dict]:
        """
        Creates a Higgs field node (PHI) and connects it to a source node.
        """
        try:
            if nid is  None:
                nid = f"PHI__{px_id}"

            node_attrs = dict(
                nid=nid,
                tid=0,
                type="HIGGS",
                px=px_id,
                parent=["HIGGS"],
            )

            return [node_attrs]
        except Exception as e:
            print(f"Err higgs_attrs: {e}")


    def higgs_params_batch(self, dim, just_vals=False, just_k=False) -> dict or list:
        #phi = self.init_phi(h)
        try:
            field_value = self.field_value(dim=dim)
            field = {
                # --- Arrays scaled by N (Structure of Arrays - SOA) ---
                "phi": field_value,
                #"prev": np.zeros_like(phi_val),  # Previous timestep state
                "dmu_h": self.dmu(dim),
                "h": field_value,  # The physical scalar field component
                "dV_dh": field_value,  # Potential derivative
                "laplacian_h": field_value,
                "vev": 246.0,  # Vacuum Expectation Value (VEV)
                "energy": 0,  # Energy contribution per node
                "energy_density": field_value,

                # --- Scalars (Constants/System Aggregates) ---
                "potential_energy_H": field_value,
                "total_energy_H": field_value,  # Total system energy (System Scalar)
                "mass": 125.0,  # Higgs mass constant
                "lambda_H": 0.13 # Coupling constant
            }

            if just_vals:
                return list(field.values())
            elif just_k:
                return list(field.keys())
            else:
                return field

        except Exception as e:
            print("Err higgs_params_batchs", e)

    def create_higgs_field(self, px_id):
        attrs = self.higgs_attrs(
            px_id
        )

        self.g.add_node(attrs)

        # Connect PHI to PIXEL
        self.g.add_edge(
            px_id,
            attrs["id"],
            attrs=dict(
                rel=f"has_field",
                src_layer="PIXEL",
                trgt_layer="PHI",
            )
        )


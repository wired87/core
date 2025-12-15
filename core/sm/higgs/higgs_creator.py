from sm.higgs.phi_utils import HiggsUtils
import jax.numpy as jnp
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
                nid = f"HIGGS__{px_id}"

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

    def higgs_params_batch(self, amount_nodes, dim, just_vals=False,just_k=False) -> dict or list:
        #h = 0.0
        N = dim
        #phi = self.init_phi(h)
        field = {
            # --- Arrays scaled by N (Structure of Arrays - SOA) ---
            "phi": self.field_values(amount_nodes, dim, distance=0),
            #"prev": jnp.zeros_like(phi_val),  # Previous timestep state
            "dmu_h": [self.dmu(amount_nodes, dim) for _ in range(len(amount_nodes))],
            "h": self.field_value("h"),  # The physical scalar field component
            "dV_dh": self.field_value("h"),  # Potential derivative
            "laplacian_h": self.field_value("h"),
            "vev": jnp.full(N, 246.0, dtype=jnp.float32),  # Vacuum Expectation Value (VEV)
            "energy": self.field_value("h"),  # Energy contribution per node
            "energy_density": self.field_value("h"),

            # --- Scalars (Constants/System Aggregates) ---
            "potential_energy_H": self.field_value("h"),  # Total potential energy (System Scalar)
            "total_energy_H": self.field_value("h"),  # Total system energy (System Scalar)
            "mass": jnp.full(N, 125.0, dtype=jnp.float32),  # Higgs mass constant
            "lambda_H": jnp.full(N, 0.13, dtype=jnp.float32)  # Coupling constant
        }

        if just_vals:
            return list(field.values())
        elif just_k:
            return list(field.keys())
        else:
            return field


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


import numpy as np
import ray

from core.app_utils import TESTING
from gpu_actor.gpu_processor import GPUProcessor
from qf_utils.field_utils import FieldUtils
from core.sm.cloupling.data_preprocessor import DataPreprocessor
import jax.numpy as jnp


@ray.remote(num_cpus=.5)
class EdgeProcessor(DataPreprocessor):

    def __init__(self):
        DataPreprocessor.__init__(self)
        self.vertex_struct = {
            "quad_vertex": self.prepare_vertex,
            "triple_vertex": self.prepare_vertex
        }
        self.g_nodes = {}

    def preprocess_edges(
            self,
            edges,
            g_nodes,
    ):
        print("========= PROCESS EDGES =========")
        self.classify_edges(edges)






    def process_vertex(self, g_nodes):
        for eq_name, eq_data_runnable in self.vertex_struct.items():
            id_map = {
                eid: 0
                for eid in g_nodes.items()
            }
            data = {}

            if "quad" in eq_name:
                # TRIPLE VERTEX
                self.prepare_vertex(g_nodes, quad=False)
                data = self.triple_vertex

            elif "triple" in eq_name:
                # QUAD VERTEX
                self.prepare_vertex(g_nodes, quad=True)
                data = self.quad_vertex

            batch_result = ray.get(
                ray.get_actor(name="EDGE_PROCESSOR").process.remote(
                    data=data,
                    eq_key=eq_name,
                )
            )

            # Apply result to id_map
            for (eid, total_val), (vid, val) in zip(id_map, batch_result):
                total_val += np.sum(val, dtype=np.float64)

            # Adapt nodes field values
            for vid in id_map:
                # Update g nodes
                nids = vid.split("___")
                for i, nid in enumerate(nids):
                    # extend field value with vertex combi
                    g_nodes[nid][self._field_value(f"{nid.split('__')[0]}")] += id_map[vid][i]

            




@ray.remote(
    num_gpus=.25 if TESTING is False else 0,
    num_cpus=.5 if TESTING is True else 0
)
class EdgeProcessorArsenal(
    FieldUtils, GPUProcessor
):

    def __init__(self):
        GPUProcessor.__init__(
            self,
            field_type="edge"
        )
        FieldUtils.__init__(self)

    def process(self, eq_key, data:dict[str, dict], id_map):
        runnable = getattr(self, eq_key)
        batch_result:list[dict] = self.calc_batch(
            list(data.values()),
            runnable
        )

        return id_map


    def triple_vertex(
            self,
            fw1,
            fw2,
            fw3,
            factor,
            V,
    ) -> list:
        for mu in range(4):
            for sigma in range(4):
                for rho in range(4):
                    term1 = self.mt[sigma, rho] * (fw1[mu] - fw2[mu])
                    term2 = self.mt[rho, mu] * (fw2[sigma] - fw3[sigma])
                    term3 = self.mt[mu, sigma] * (fw3[rho] - fw1[rho])
                    V[mu, sigma, rho] = factor * (term1 + term2 + term3)


    def constructor(self, factor):
        """
        Berechnet den Quartic Vertex Tensor V[mu, nu, sigma, rho]
        für W- W+ Photon Photon.
        """
        V = jnp.zeros((4, 4, 4, 4), dtype=complex)
        for mu in range(4):
            for nu in range(4):
                for sigma in range(4):
                    for rho in range(4):
                        term1 = 2 * self.mt[sigma, rho] * self.mt[mu, nu]
                        term2 = -self.mt[sigma, mu] * self.mt[rho, nu]
                        term3 = -self.mt[sigma, nu] * self.mt[rho, mu]
                        V[mu, nu, sigma, rho] = factor * (term1 + term2 + term3)
        return V


    def quad_vertex(
            self,
            fw1,
            fw2,
            fw3,
            fw4,
            factor,
    ):
        j_nu = jnp.zeros(4)
        constructor = self.constructor(factor)
        for mu in range(4):
            j_sum = 0
            for nu in range(4):
                for sigma in range(4):
                    for rho in range(4):
                        j_sum += constructor[mu, nu, sigma, rho] * fw1[mu] * fw2[nu] * fw3[sigma] * fw4[rho]
            j_nu[mu] += j_sum
        return j_nu


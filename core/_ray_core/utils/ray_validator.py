import ray
from app_utils import GLOBAC_STORE


class RayValidator:
    def __init__(self, host=None, g_utils=None, g=None):
        self.host = host
        if self.host is None:
            from qf_utils.calculator.calculator import Calculator
            from qf_utils.qf_utils import QFUtils
            self.g_utils = g or g_utils
            self.qfu = QFUtils(
                self.g_utils,
            )
            self.calculator = Calculator()

    def call(self, method_name, *args, **kwargs):
        if self.host is not None:
            result = ray.get(
                GLOBAC_STORE["UTILS_WORKER"].call.remote(
                    method_name=method_name,
                    *args,
                    **kwargs
                )
            )
        else:
            result = self.call_local(
                method_name, *args, **kwargs
            )
        return result

    def call_local(self, method_name, *args, **kwargs):
        print("self.gggggggggg_utils", self.g_utils)
        if hasattr(self.g_utils, method_name):
            return getattr(self.g_utils, method_name)(*args, **kwargs)
        elif hasattr(self.qfu, method_name):
            return getattr(self.qfu, method_name)(*args, **kwargs)
        elif hasattr(self.calculator, method_name):
            return getattr(self.calculator, method_name)(*args, **kwargs)
        else:
            print(f"Method {method_name} does not exists in g or qfu")

    def get_neighbor(self, nid, trgt_rel=None, trgt_type="PHI", single=True):
        if self.host is not None:
            if single is True:
                return ray.get(GLOBAC_STORE["UTILS_WORKER"].get_neighbor.remote(
                    just_id=True,
                    single=single
                ))

        else:
            if single is True:
                return self.g_utils.get_single_neighbor_nx(nid, trgt_type)
            else:
                return self.g_utils.get_neighbor_list(nid, trgt_rel=trgt_rel)

    def get_node(self, nid):
        if self.host is None:
            return self.g_utils.G.nodes[nid]
        else:
            ray.get(GLOBAC_STORE["UTILS_WORKER"].get_node.remote(nid))

    def update_edge(self, src, trgt, attrs, rels):
        if self.host is not None:
            ray.get(GLOBAC_STORE["UTILS_WORKER"].update_edge.remote(
                src,
                trgt,
                attrs,
                rels
            ))
        else:
            self.g_utils.update_edge(
                src,
                trgt,
                attrs,
                rels
            )

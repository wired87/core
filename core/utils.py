

class Preprocessor:

    def __init__(self, host, env):
        self.data_processors = {
            "FERMION": {
                "_coupling_term": None  # space for processor action
            }
        }
        self.env = env
        self.host = host
        self.utils = {
            "env",
            "host",
            "neighbor_pm_val_same_type",
            "all_subs",
            "neighbor_pm_val_fmunu",
            "self_item_up_path",
            "self_h_entry_item_up_path",
            "neighbor_node_ids",
            "edge_ids",
            "parent_pixel_id",
        }



    def get_env(self):
        return self.env

    def process(self, utils_keys, attr_struct: list):
        for attrs in attr_struct:
            rt_utils = self.ruc.create_node_rtu(**attrs)
            if rt_utils is not None:
                node_rtu = self.ruc.create_node_rtu_filtered(
                    self.host,
                    attrs,
                    env=self.env,
                    extractor_keys=utils_keys
                )
                attrs.update(node_rtu)
        return attr_struct


    def check_preprocess(self, eq_key):
        if eq_key in self.data_processors:
            return True
        return False




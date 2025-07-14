import itertools


class StimUtils:


    def __init__(self, node_ids):
        self.node_combi_generator = self._node_combi_generator(node_ids)




    def next_node_combi(self):
        return next(self.node_combi_generator)

    def _node_combi_generator(self, node_ids):
        param_lists = []
        for node in node_ids:
            phases = list(range(1, 101))
            strengths = [round(x * 0.1, 1) for x in range(0, 101)]
            param_lists.append(itertools.product(phases, strengths))

        for combo in itertools.product(*param_lists):
            combo_entry = {}
            for node_id, (phase, strength) in zip(node_ids, combo):
                combo_entry[node_id] = {
                    "phase": phase,
                    "strength": strength
                }
            yield combo_entry
from typing import NamedTuple
import jax.numpy as jnp





@ray.remote
class GNNCreator:


    def __init__(self, g):
        self.g = g


    def create_nodes(self, field_map):
        print("create_nodes started")
        for f in field_map:
            ref = GNode.options(
                name=f,
                lifetime="detached"
            ).remote()

            self.g.update_node(dict(
                nid=f,
                ref=ref,
            ))
        print("create_nodes finished")




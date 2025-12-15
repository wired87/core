"""



"""
from dataclasses import dataclass
from typing import List
import jax.numpy as jnp

@dataclass
class GStore:
    nodes: List[jnp.ndarray]   # one array per module: shape (N_nodes_module, feat_dim)
    edges: List[jnp.ndarray]   # optional adjacency / edge features
    inj_pattern: List
    method_struct:List



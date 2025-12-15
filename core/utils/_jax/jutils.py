from jax import lax

import jax.numpy as jnp


class JUtils:

    def __init__(self):
        pass

    def gather_items(self, grid, idx_list):
        idx_list = jnp.asarray(idx_list, dtype=jnp.int32)

        return lax.gather(
            grid,
            idx_list,
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0, 1, 2),
                start_index_map=(0, 1, 2),
            ),
            slice_sizes=(1, 1, 1)
        ).reshape(-1)
    
    
    def convert_type_jnp(self, const, k, struct:dict):
        print("convert_type_jnp:", k, type(const))
        try:
            if isinstance(const, (list, tuple)):
                struct[k] = const

            elif isinstance(const, int):
                struct[k] = jnp.int64(const)

            elif isinstance(const, float):
                struct[k] = jnp.float64(const)

            elif isinstance(const, complex):
                struct[k] = jnp.complex64(const)

            else:
                print("Err", const)
                struct[k] = const
        except Exception as e:
            print("Err convert_type_jnp:", e)

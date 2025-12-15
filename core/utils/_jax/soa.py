
import jax.numpy as jnp

def list_to_soa(attrs_list: list[dict]) -> dict[str, jnp.ndarray]:
    # Utility to convert List[Dict] to Struct-of-Arrays (SOA)
    if not attrs_list: return {}
    # In a real scenario, this involves complex deserialization and array alignment
    return {k: jnp.array([d[k] for d in attrs_list]) for k in attrs_list[0]}


def soa_to_list(attrs_soa: dict[str, jnp.ndarray]) -> list[dict]:
    if not attrs_soa: return []
    num_fields = len(next(iter(attrs_soa.values())))
    attrs_list = []
    for i in range(num_fields):
        attrs_list.append({k: v[i].tolist() if hasattr(v[i], 'tolist') else v[i] for k, v in attrs_soa.items()})
    return attrs_list



def dict_to_soa(
        struct: dict[str, dict]
) -> dict[str, jnp.ndarray]:
    soa = {
        key: jnp.array(
            [
                gattrs[key]
                for gattrs in list(struct.values())
            ]
        )
        for key, attrs in struct.items()
    }
    return soa
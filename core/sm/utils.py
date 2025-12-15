from jax import jit, vmap


@jit
def _dmu(field_forward, field_backward, d):
    """Calculates single d(phi)/dx element-wise."""
    return (field_forward - field_backward) / (2.0 * d)


@jit
def dX(
    field_value,
    prev,
    d,
    t,
    neighbor_pm_val_same_type,
    _dmu
):
    """
    Kinetisch ableitung
    """
    # 📌 Die Dirac-Gleichung kombiniert alle berechneten Kopplungsterme
    dt = (field_value - prev) / t

    def wrapper(attrs):
        """
        val_plus,
        val_minus,
        d
        """
        dmu_x = _dmu(
            *attrs,
            d
        )
        return dmu_x

    vmapped_func = vmap(wrapper, in_axes=(0, None))

    # Static arguments are passed explicitly using static_argnums
    compiled_kernel = jit(vmapped_func)  # , static_argnums=(1,)

    calc_result = compiled_kernel(neighbor_pm_val_same_type)
    dmu = [dt, *calc_result]
    return dmu
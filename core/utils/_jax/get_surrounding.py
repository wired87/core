from jax import vmap
from ray.rllib.utils.typing import jnp


def get_surrounding(world):

    def _calc_spatial_derivative(dir_p, dir_m):
        # np und nm sind die VOLLSTÄNDIGEN verschobenen Gitter-Arrays
        np = jnp.roll(world, shift=dir_p, axis=(0, 1, 2, 3))
        nm = jnp.roll(world, shift=dir_m, axis=(0, 1, 2, 3))

        # Die Berechnung wird Array-weise durchgeführt.
        calc_result = _d_spatial(
            field_forward=np,
            field_backward=nm,
            d_space=d
        )
        return calc_result

    # VMAP iteriert korrekt über die Listen der Richtungs-Tupel (pm_axes)
    vmapped_func = vmap(
        _calc_spatial_derivative,
        in_axes=(0, 0)
    )

    # JIT-Kompilierung des VMAP-Kerns
    dmu_spatial = vmapped_func(*pm_axes)


@jit
def dmuG(
        world,
        field_value,  # Aktuelles Gitter (Psi(t))
        prev,  # Gitter des vorherigen Zeitschritts (Psi(t-dt))
        d,  # Räumlicher Gitterabstand d
        dt,  # Zeitschrittweite
        pm_axes: tuple[list[tuple], list[tuple]],  # ([+dirs], [-dirs])
):
    """
    Kinetische Ableitung (d/dt und 26 verallgemeinerte räumliche Ableitungen).
    Verwendet Central Difference (1. Ableitung) für die räumlichen Terme.
    """

    # 1. Funktion für die räumliche Zentraldifferenz
    @jit
    def _d_spatial(field_forward, field_backward, d_space):
        """Berechnet (Psi_{i+1} - Psi_{i-1}) / (2.0 * d) für das gesamte Array."""
        return (field_forward - field_backward) / (2.0 * d_space)

    # 2. Funktion für die zeitliche Ableitung (Backward Difference)
    def _d_time(field_current, field_prev, d_time):
        """Berechnet (Psi(t) - Psi(t-dt)) / dt für das gesamte Array."""
        return (field_current - field_prev) / d_time

    # --- Räumliche Ableitungen (Batch-Verarbeitung der 13 Paare) ---



    # JIT-Kompilierung des VMAP-Kerns
    dmu_spatial = vmapped_func(*pm_axes)  # Liefert ein Array von 13 Ableitungs-Arrays

    # --- Zeitliche Ableitung ---
    time_res = _d_time(
        field_current=field_value,
        field_prev=prev,
        d_time=dt
    )

    # Konvertierung des Array-Ergebnisses des vmap-Kerns in eine Liste von Arrays
    spatial_list = list(dmu_spatial)

    # Zeitlichen Term an erster Stelle einfügen
    spatial_list.insert(0, time_res)

    return spatial_list
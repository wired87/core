"""
Modular visualizer: adaptable to any cfg specifications and data keys.
Schema remains same; keys are configurable.

Generates plots of all data keys within env_cfg.dims-dimensional space
(white background, light blue data points) for each time step, rendering
all data points of the current 1d scaled db (time_db[0]) using shape-,
param- and other controllers for high JAX operation performance.
"""

from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


# Default schema: cfg key names for controllers and env.
# Override schema keys to adapt to different cfg specifications; structure stays same.
DEFAULT_SCHEMA = {
    "dims": "dims",
    "dims_alt": "DIMS",
    "amount_nodes": "amount_of_nodes",
    "amount_nodes_alt": "AMOUNT_NODES",
    "param_ctrl": "DB_PARAM_CONTROLLER",
    "amount_per_field": "AMOUNT_PARAMS_PER_FIELD",
    "db_shape": "DB_SHAPE",
    "modules": "MODULES",
    "fields": "FIELDS",
    "db_keys": "DB_KEYS",
    "field_keys": "FIELD_KEYS",
}


def _get_positions(amount: int, dims: int) -> list[tuple[int, ...]]:
    """Grid positions in dims-dimensional space."""
    return list(product(range(amount), repeat=dims))


def _reshape_flat_db(
    flat: np.ndarray,
    param_ctrl: list[int],
    amount_per_field: list[int],
    n_modules: int,
    n_fields: int,
) -> list[list[list[np.ndarray]]]:
    """
    Reshape 1d flat db into [modules][fields][params] using controllers.
    JAX-friendly: returns contiguous arrays for each param.
    """
    offset = 0
    ctrl_idx = 0
    out = []
    for mi in range(n_modules):
        mod_data = []
        for fi in range(n_fields):
            flat_idx = mi * n_fields + fi
            n_params = amount_per_field[flat_idx] if flat_idx < len(amount_per_field) else 1
            field_data = []
            for _ in range(n_params):
                n_vals = param_ctrl[ctrl_idx] if ctrl_idx < len(param_ctrl) else 1
                ctrl_idx += 1
                end = offset + n_vals
                if end <= len(flat):
                    chunk = np.asarray(flat[offset:end], dtype=np.float32)
                else:
                    chunk = np.zeros(n_vals, dtype=np.float32)
                field_data.append(chunk)
                offset = end
            mod_data.append(field_data)
        out.append(mod_data)
    return out


class ModularVisualizer:
    """
    Modular visualizer adaptable to any cfg specifications and data keys.
    Schema remains the same; key names are configurable via schema dict.

    Renders all data points of time_db[time] in env_cfg.dims-dimensional space
    using shape-, param- and other controllers.

    Attributes:
        schema: dict mapping logical names to cfg keys (default: DEFAULT_SCHEMA)
        env_cfg: env config with dims, amount_of_nodes
        cfg: full components cfg (DB_PARAM_CONTROLLER, etc.)
    """

    def __init__(
        self,
        env_cfg: dict[str, Any],
        cfg: dict[str, Any],
        schema: dict[str, str] | None = None,
    ):
        self.env_cfg = env_cfg or {}
        self.cfg = cfg or {}
        self.schema = {**DEFAULT_SCHEMA, **(schema or {})}

    def _resolve(self, key: str) -> Any:
        """Resolve value from env_cfg or cfg using schema key."""
        sk = self.schema.get(key, key)
        if sk in self.env_cfg:
            return self.env_cfg[sk]
        if sk in self.cfg:
            return self.cfg[sk]
        return None

    def _get_dims(self) -> int:
        return int(self._resolve("dims") or self._resolve("dims_alt") or 3)

    def _get_amount_nodes(self) -> int:
        n = self._resolve("amount_nodes")
        if n is None:
            n = self._resolve("amount_nodes_alt")
        return int(n or 1)

    def _get_positions(self) -> list[tuple[int, ...]]:
        return _get_positions(self._get_amount_nodes(), self._get_dims())

    def _reshape_time_db(self, time_db_flat: np.ndarray) -> list[list[list[np.ndarray]]]:
        """Reshape time_db[0] (1d) using controllers."""
        param_ctrl = self.cfg.get(self.schema["param_ctrl"]) or []
        amount_per_field = self.cfg.get(self.schema["amount_per_field"]) or []
        modules = self.cfg.get(self.schema["modules"]) or [0]
        fields = self.cfg.get(self.schema["fields"]) or [1]

        n_modules = max(1, max(modules) + 1) if modules else 1
        n_fields = max(1, max(fields)) if fields else 1

        flat = np.asarray(time_db_flat)
        if np.iscomplexobj(flat):
            flat = np.abs(flat.astype(np.complex64)).astype(np.float32)
        else:
            flat = flat.astype(np.float32)

        return _reshape_flat_db(
            flat, param_ctrl, amount_per_field, n_modules, n_fields
        )

    def _extract_data_points(
        self,
        reshaped: list[list[list[np.ndarray]]],
        positions: list[tuple[int, ...]],
    ) -> np.ndarray:
        """
        Extract one scalar per position for plotting.
        Flattens all param arrays in controller order; takes first len(positions) values.
        """
        points = []
        for mod in reshaped:
            for field in mod:
                for param_arr in field:
                    for v in param_arr.flat:
                        points.append(float(np.real(v)) if np.iscomplexobj(v) else float(v))
        n_pos = len(positions)
        if not points:
            return np.zeros(n_pos, dtype=np.float32)
        arr = np.array(points[:n_pos], dtype=np.float32)
        if len(arr) < n_pos:
            arr = np.pad(arr, (0, n_pos - len(arr)), constant_values=0.0)
        return arr

    def _extract_positions_and_values(
        self,
        time_db_flat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (positions, values) for plotting from time_db[0]."""
        positions = self._get_positions()
        reshaped = self._reshape_time_db(time_db_flat)
        values = self._extract_data_points(reshaped, positions)

        pos_arr = np.array(positions, dtype=np.float32)
        if pos_arr.size == 0 or pos_arr.ndim == 1:
            dims = self._get_dims()
            pos_arr = np.zeros((len(positions), max(1, dims)), dtype=np.float32)
        return pos_arr, values

    def plot_time_step(
        self,
        time_db: list[np.ndarray] | np.ndarray,
        time_idx: int = 0,
        ax=None,
        **kwargs,
    ):
        """
        Plot all data points for time step time_idx from time_db[time_idx].

        Args:
            time_db: list of 1d arrays (time_db[0] = current step)
            time_idx: index into time_db
            ax: matplotlib axes (optional)
            **kwargs: passed to scatter (e.g. s=, alpha=)
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for ModularVisualizer.plot_time_step")

        if isinstance(time_db, np.ndarray):
            time_db = [time_db]
        flat = time_db[time_idx] if time_idx < len(time_db) else time_db[0]

        pos_arr, values = self._extract_positions_and_values(flat)
        dims = self._get_dims()

        scatter_kw = {"c": "#add8e6", "edgecolors": "none", **kwargs}

        if ax is None:
            fig = plt.figure(facecolor="white")
            if dims >= 3:
                ax = fig.add_subplot(111, projection="3d", facecolor="white")
            else:
                ax = fig.add_subplot(111, facecolor="white")

        if dims == 1:
            x = np.arange(len(values))
            ax.scatter(x, values, **scatter_kw)
        elif dims == 2:
            x, y = pos_arr[:, 0], pos_arr[:, 1]
            ax.scatter(x, y, **scatter_kw)
        else:
            x, y, z = pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2]
            ax.scatter(x, y, z, **scatter_kw)

        ax.set_facecolor("white")
        ax.patch.set_facecolor("white")
        if ax.figure:
            ax.figure.patch.set_facecolor("white")
        return ax

    def render_all_time_steps(
        self,
        time_db: list[np.ndarray],
        out_path: str | None = None,
        **kwargs,
    ) -> list[Any]:
        """
        Render all time steps. Returns list of axes.
        If out_path: save as {out_path}_t{idx}.png per step.
        """
        if not _HAS_MATPLOTLIB:
            raise ImportError("matplotlib required for render_all_time_steps")

        axes = []
        for t in range(len(time_db)):
            fig = plt.figure(facecolor="white")
            ax = fig.add_subplot(111, facecolor="white")
            self.plot_time_step(time_db, time_idx=t, ax=ax, **kwargs)
            axes.append(ax)

            if out_path:
                base = out_path.rsplit(".", 1)[0] if "." in out_path else out_path
                ext = out_path.rsplit(".", 1)[-1] if "." in out_path else "png"
                plt.savefig(f"{base}_t{t}.{ext}", facecolor="white", bbox_inches="tight")
                plt.close()

        return axes

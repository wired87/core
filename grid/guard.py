"""
Grid Guard: consumes cfg (components from core.guard.converter) and runs the expected workflow.
Decodes DB, builds model structure, and saves the final model artifact.
"""

import base64
import json
import os
from typing import Any

import numpy as np


class Guard:
    """
    Consumes cfg (components dict) and runs the grid workflow.
    Cfg is the output of core.guard.Guard.converter() (e.g. from test_out.json).
    """

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}

    def run(self, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        Run the expected workflow with the given cfg.
        If cfg is provided, use it; otherwise use self.cfg.
        """
        cfg = cfg if cfg is not None else self.cfg
        if not cfg:
            print("[grid.Guard] No cfg provided, skipping workflow")
            return {}

        self._validate_cfg(cfg)
        return self._run_workflow(cfg)

    def _validate_cfg(self, cfg: dict) -> None:
        """Validate minimal structure required for grid-root."""
        required = ("DB", "AXIS", "FIELDS", "MODULES")
        for k in required:
            if k not in cfg:
                raise ValueError(f"[grid.Guard] cfg missing required key: {k}")

    def _decode_db(self, cfg: dict) -> np.ndarray | None:
        """Decode DB from base64 to numpy array (complex64)."""
        db_b64 = cfg.get("DB")
        if not db_b64:
            return None
        try:
            raw = base64.b64decode(db_b64)
            return np.frombuffer(raw, dtype=np.complex64)
        except Exception as e:
            print(f"[grid.Guard] DB decode warning: {e}")
            return None

    def _build_nodes_from_cfg(self, cfg: dict, db_flat: np.ndarray) -> list:
        """
        Build nodes structure [modules][fields][params] -> array of values.
        Uses DB_PARAM_CONTROLLER (len per param), AMOUNT_PARAMS_PER_FIELD (params per field).
        """
        param_ctrl = cfg.get("DB_PARAM_CONTROLLER") or []
        amount_per_field = cfg.get("AMOUNT_PARAMS_PER_FIELD") or []
        modules = cfg.get("MODULES") or [0]
        fields = cfg.get("FIELDS") or [1]

        n_modules = max(1, max(modules) + 1) if modules else 1
        n_fields = max(1, max(fields)) if fields else 1

        offset = 0
        ctrl_idx = 0
        nodes_list = []
        for mi in range(n_modules):
            mod_nodes = []
            for fi in range(n_fields):
                flat_idx = mi * n_fields + fi
                n_params = amount_per_field[flat_idx] if flat_idx < len(amount_per_field) else 1
                field_nodes = []
                for _ in range(n_params):
                    n_vals = param_ctrl[ctrl_idx] if ctrl_idx < len(param_ctrl) else 1
                    ctrl_idx += 1
                    end = offset + n_vals
                    if end <= len(db_flat):
                        chunk = db_flat[offset:end].copy()
                    else:
                        chunk = np.zeros(n_vals, dtype=np.complex64)
                    field_nodes.append(chunk)
                    offset = end
                mod_nodes.append(field_nodes)
            nodes_list.append(mod_nodes)

        return nodes_list

    def _run_workflow(self, cfg: dict) -> dict:
        """
        Execute the grid workflow: decode DB, build model, save artifact.
        """
        print("[grid.Guard] Running workflow with cfg keys:", list(cfg.keys()))

        db_flat = self._decode_db(cfg)
        model_data = {"config": cfg}

        if db_flat is not None:
            nodes = self._build_nodes_from_cfg(cfg, db_flat)
            model_data["db_decoded_len"] = int(len(db_flat))
            model_data["structure"] = {
                "n_modules": len(nodes),
                "n_fields": len(nodes[0]) if nodes else 0,
            }

        out_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.getenv("GRID_MODEL_OUT", os.path.join(out_dir, "model_out.json"))

        def _json_serial(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj) if isinstance(obj, np.floating) else int(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(model_path, "w", encoding="utf-8") as f:
            json.dump(model_data, f, indent=2, default=_json_serial)

        print(f"[grid.Guard] Model artifact written to {model_path}")

        npz_path = model_path.replace(".json", "_data.npz")
        if db_flat is not None:
            np.savez_compressed(npz_path, db_flat=db_flat)
            print(f"[grid.Guard] Model data written to {npz_path}")

        return model_data

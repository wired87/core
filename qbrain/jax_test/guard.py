import base64
import json
import os
from typing import Any, Dict, List

import jax.numpy as jnp

import jax
from qbrain.jax_test.jax_utils.deserialize_in import parse_value
from qbrain.auth.load_sa_creds import load_service_account_credentials
from qbrain.jax_test.data_handler.main import load_data
from qbrain.jax_test.gnn.gnn import GNN


def _to_json_serializable(data):
    """Convert JAX/numpy arrays and complex to JSON-serializable (list/dict)."""
    if isinstance(data, list):
        return [_to_json_serializable(x) for x in data]
    if isinstance(data, tuple):
        return [_to_json_serializable(x) for x in data]
    if isinstance(data, dict):
        return {k: _to_json_serializable(v) for k, v in data.items()}
    if isinstance(data, bytes):
        # --- FIX: bytes (e.g. from base64.b64encode) -> decode to string for JSON ---
        # base64.b64encode returns ASCII-safe bytes, so decode as ASCII
        return data.decode('ascii')
    if hasattr(data, "shape") and hasattr(data, "tolist"):
        arr = jnp.asarray(data)
        if jnp.iscomplexobj(arr):
            return {"real": jnp.real(arr).tolist(), "imag": jnp.imag(arr).tolist()}
        return arr.tolist()
    if hasattr(data, "item"):
        v = data.item()
        if isinstance(v, (complex, jnp.complexfloating)):
            return {"real": float(jnp.real(v)), "imag": float(jnp.imag(v))}
        return float(v) if hasattr(v, "real") else v
    return data


def _sanitize_param_column_name(raw_id: str) -> str:
    """
    Map a param id to a safe column name for the envs table.
    Mirrors logic from ParamsManager._sanitize_param_column_name.
    """
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(raw_id))
    if not safe:
        safe = "param"
    if safe[0].isdigit():
        safe = f"p_{safe}"
    return f"p_{safe}"


class Guard:
    # todo prevaliate features to avoid double calculations
    def __init__(self):
        #JAX
        platform = "cpu" if os.name == "nt" else "gpu"
        jax.config.update("jax_platform_name", platform)  # must be run before jnp
        self.gpu = jax.devices(platform)[0]

        # LOAD DAT FROM BQ OR LOCAL
        self.cfg = load_data()

        AMOUNT_NODES = int(os.getenv("AMOUNT_NODES"))
        SIM_TIME = int(os.getenv("SIM_TIME"))
        DIMS = int(os.getenv("DIMS"))

        for k, v in self.cfg.items():
            self.cfg[k] = parse_value(v)

            if isinstance(self.cfg[k], dict):
                for i, o in self.cfg[k].items():
                    self.cfg[k][i] = parse_value(o)

        # layers
        self.gnn_layer = GNN(
            amount_nodes=AMOUNT_NODES,
            time=SIM_TIME,
            gpu=self.gpu,
            DIMS=DIMS,
            **self.cfg
        )

        self._live_ws = None

    def divide_vector(self, vec, divisor):
        """Divide all values of a given vector by divisor. Returns array same shape as vec."""
        v = jnp.asarray(vec)
        d = jnp.asarray(divisor)
        return v / d

    def _make_send_live(self, vis_fps):
        """Return a callable (gnn_self, step) -> None that sends LIVE_DATA via self._live_ws."""
        import numpy as np
        from grid.live_payload import build_live_data_payload

        ws = self._live_ws
        user_id = os.getenv("USER_ID")
        env_id = os.getenv("ENV_ID")
        if not ws or not user_id or not env_id:
            return None

        def send_live(gnn_self, step):
            if ws is None:
                return
            # Throttle: send roughly every N steps to cap frame rate
            counter = getattr(gnn_self, "_vis_step_counter", 0)
            gnn_self._vis_step_counter = counter + 1
            n = max(1, 100 // max(1, vis_fps))
            if counter % n != 0:
                return
            try:
                dl = gnn_self.db_layer
                flat = np.asarray(dl.time_construct[0]).ravel()
                if np.iscomplexobj(flat):
                    flat = np.abs(flat.astype(np.complex64)).astype(np.float32)
                cfg = {
                    k: getattr(gnn_self, k, None)
                    for k in ("DB_PARAM_CONTROLLER", "AMOUNT_PARAMS_PER_FIELD", "MODULES", "FIELDS", "DB_KEYS", "FIELD_KEYS")
                }
                data = build_live_data_payload(cfg, flat)
                payload = {
                    "type": "LIVE_DATA",
                    "auth": {"user_id": user_id, "env_id": env_id},
                    "data": data,
                }
                ws.send(json.dumps(payload, default=str))
            except Exception as e:
                print("[jax_test.Guard] LIVE_DATA send error:", e)

        return send_live

    def main(self):
        self.gnn_layer.main()
        #results = self.finish()
        print("SIMULATION PROCESS FINISHED")
        return None


    def run(self):

        print("run... done")



    def _export_data(self):
        print("_export_data...")
        dl = self.gnn_layer.db_layer

        history = []
        env_id = os.getenv("ENV_ID")

        for i, item in enumerate(dl.history_nodes):
            history.append(
                {
                    "id": f"{env_id}_{i}",
                    "data":_to_json_serializable(item),
                    "env_id": env_id
                }
            )

        # INSERT
        self.bqclient.bq_insert(
            table_id="data",
            rows = history
        )
        print("_export_data... done")

    def _export_ctlr(self):
        print("_export_ctlr...")
        dl = self.gnn_layer.db_layer
        env_id = os.getenv("ENV_ID")

        db_ctlr = {
            "id": env_id,
            "OUT_SHAPES": _to_json_serializable(dl.OUT_SHAPES),
            "SCALED_PARAMS": _to_json_serializable(dl.SCALED_PARAMS),
            "METHOD_TO_DB": _to_json_serializable(dl.METHOD_TO_DB),
            "AMOUNT_PARAMS_PER_FIELD": _to_json_serializable(dl.AMOUNT_PARAMS_PER_FIELD),
            "DB_PARAM_CONTROLLER": _to_json_serializable(dl.DB_PARAM_CONTROLLER),
            "DB_KEYS": _to_json_serializable(self.cfg["DB_KEYS"]),
            "FIELD_KEYS": _to_json_serializable(self.cfg["FIELD_KEYS"])
        }


        model_ctlr = {
            "id": env_id,
            "VARIATION_KEYS": _to_json_serializable(self.cfg["VARIATION_KEYS"]),
        }



        # INSERT
        self.bqclient.bq_insert(
            table_id="data",
            rows = [db_ctlr]
        )
        print("_export_ctlr... done")



    def _export_engine_state(self, serialized_in, serialized_out, out_path: str = "engine_output.json"):
        """Save all generated engine data (history, db, tdb, etc.) to a local .json file."""
        dl = self.gnn_layer.db_layer
        try:
            payload = {
                "serialized_out": _to_json_serializable(base64.b64encode(serialized_out).decode('ascii')),
                "serialized_in": _to_json_serializable(base64.b64encode(serialized_in).decode('ascii')),

                # CTLR
                "ENERGY_MAP": None,
            }
            if hasattr(dl, "tdb") and dl.tdb is not None:
                payload["tdb"] = _to_json_serializable(dl.tdb)

            self._upsert_generated_data_to_bq(payload)

            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, allow_nan=True)
            print("engine state saved to", out_path)
        except Exception as e:
            print("export_engine_state failed:", e)

        # Persist stacked param series (values + features) into envs table.
        try:
            self._persist_param_series_to_env(dl)
        except Exception as e:
            print("[jax_test.Guard] persist_param_series_to_env warning:", e)

    def _build_param_keys(self) -> List[str]:
        """
        Build a flat list of param keys in the same order as controller metadata.
        Mirrors jax_test.grid.live_payload._flat_keys_from_cfg.
        """
        gnn = self.gnn_layer
        param_ctrl = getattr(gnn, "DB_PARAM_CONTROLLER", []) or []
        amount_per_field = getattr(gnn, "AMOUNT_PARAMS_PER_FIELD", []) or []
        modules = getattr(gnn, "MODULES", None) or [0]
        fields = getattr(gnn, "FIELDS", None) or [1]
        db_keys = getattr(gnn, "DB_KEYS", None)
        field_keys = getattr(gnn, "FIELD_KEYS", None)

        try:
            param_ctrl = list(param_ctrl)
        except TypeError:
            param_ctrl = [param_ctrl]
        try:
            amount_per_field = list(amount_per_field)
        except TypeError:
            amount_per_field = [amount_per_field]
        try:
            modules = list(modules)
        except TypeError:
            modules = [modules]
        try:
            fields = list(fields)
        except TypeError:
            fields = [fields]

        n_modules = max(1, max(modules) + 1) if modules else 1
        n_fields = max(1, max(fields)) if fields else 1

        keys: List[str] = []
        idx = 0
        for _mi in range(n_modules):
            for _fi in range(n_fields):
                flat_idx = _mi * n_fields + _fi
                n_params = amount_per_field[flat_idx] if flat_idx < len(amount_per_field) else 1
                for _pi in range(n_params):
                    if db_keys and idx < len(db_keys):
                        keys.append(str(db_keys[idx]))
                    elif field_keys and idx < len(field_keys):
                        keys.append(str(field_keys[idx]))
                    else:
                        keys.append(f"p_{idx}")
                    idx += 1
        return keys

    def _build_param_series_payload(self, dl) -> Dict[str, Any]:
        """
        Combine DBLayer per-param histories into column payloads:
            { env_col_name: {\"values\": [...], \"features\": [...] }, ... }
        """
        values_hist: Dict[int, list] = getattr(dl, "param_values_history", {}) or {}
        features_hist: Dict[int, list] = getattr(dl, "param_features_history", {}) or {}
        if not values_hist and not features_hist:
            return {}

        keys = self._build_param_keys()
        all_indices = sorted(set(list(values_hist.keys()) + list(features_hist.keys())))
        out: Dict[str, Any] = {}
        for idx in all_indices:
            if idx < 0:
                continue
            key = keys[idx] if idx < len(keys) else f"p_{idx}"
            col_name = _sanitize_param_column_name(key)
            vals = values_hist.get(idx, []) or []
            feats = features_hist.get(idx, []) or []
            # Ensure JSON-serializable primitives.
            vals_f = [float(v) for v in vals]
            feats_f = [float(f) for f in feats]
            out[col_name] = {
                "values": vals_f,
                "features": feats_f,
            }
        return out

    def _persist_param_series_to_env(self, dl) -> None:
        """
        Persist merged param series into envs table row for (env_id, user_id, goal_id).
        """
        try:
            from qbrain.core.env_manager.env_lib import EnvManager
        except Exception as e:
            print("[jax_test.Guard] EnvManager import warning:", e)
            return

        env_id = os.getenv("ENV_ID")
        user_id = os.getenv("USER_ID")
        goal_id = os.getenv("GOAL_ID")

        if not env_id or not user_id:
            print("[jax_test.Guard] _persist_param_series_to_env: missing ENV_ID or USER_ID, skipping")
            return

        series = self._build_param_series_payload(dl)
        if not series:
            print("[jax_test.Guard] _persist_param_series_to_env: empty series, skipping")
            return

        try:
            mgr = EnvManager()
            mgr.update_env_param_series(env_id=env_id, user_id=user_id, goal_id=goal_id, param_series=series)
        except Exception as e:
            print("[jax_test.Guard] _persist_param_series_to_env: update_env_param_series warning:", e)

    def finish(self):
        # Collect data
        history_nodes = self.gnn_layer.db_layer.history_nodes
        model_skeleton = self.gnn_layer.db_layer.model_skeleton

        # Serialization helper
        def serialize(data):
            if isinstance(data, list):
                return [serialize(x) for x in data]
            if isinstance(data, tuple):
                return tuple(serialize(x) for x in data)
            if isinstance(data, dict):
                 return {k: serialize(v) for k, v in data.items()}

            # Check for JAX/Numpy array
            if hasattr(data, 'dtype') and hasattr(data, 'real') and hasattr(data, 'imag'):
                # Check directly if complex dtype
                if jnp.iscomplexobj(data):
                    return (data.real, data.imag)
            return data

        serialized_history = serialize(history_nodes)
        serialized_model = serialize(model_skeleton)

        # Construct result dictionary
        result = {
            "DB_CONTROLLER": serialized_history,
            "MODEL_CONTROLLER": serialized_model
        }

        # --- BQ UPSERT: bytified model + full history + controller meta into BigQuery ---
        try:
            from google.cloud import bigquery
            from google.api_core.exceptions import NotFound
        except ImportError as e:
            print("BigQuery client not available, skipping upsert:", e)
            print("DATA DISTRIBUTED")
            return result

        project = os.getenv("BQ_PROJECT")
        dataset = os.getenv("DS")
        table = os.getenv("TABLE")
        model_col = os.getenv("MODEL_COL")
        data_col = os.getenv("DATA_COL")
        ctlr_col = os.getenv("CTLR_COL")

        if not (project and dataset and table and model_col and data_col and ctlr_col):
            print("BigQuery env vars missing (BQ_PROJECT/DS/TABLE/MODEL_COL/DATA_COL/CTL_RCOL), skipping upsert.")
            print("DATA DISTRIBUTED")
            return result

        table_id = f"{project}.{dataset}.{table}"

        try:
            creds = load_service_account_credentials(
                scopes=["https://www.googleapis.com/auth/bigquery"]
            )
            client = bigquery.Client(project=project, credentials=creds)
        except Exception as e:
            print("Failed to initialize BigQuery client, skipping upsert:", e)
            print("DATA DISTRIBUTED")
            return result

        # Bytify model, history (all time steps), and controller metadata
        try:
            model_bytes = self.gnn_layer.serialize(model_skeleton)
            history_bytes = self.gnn_layer.serialize(history_nodes)
            ctrl_payload = {
                "cfg": self.cfg,
                "AMOUNT_NODES": int(os.getenv("AMOUNT_NODES", "0") or 0),
                "SIM_TIME": int(os.getenv("SIM_TIME", "0") or 0),
                "DIMS": int(os.getenv("DIMS", "0") or 0),
            }
            ctrl_bytes = self.gnn_layer.serialize(ctrl_payload)
        except Exception as e:
            print("Failed to bytify model/history/controller, skipping upsert:", e)
            print("DATA DISTRIBUTED")
            return result

        # Ensure target table with BYTES columns exists
        try:
            client.get_table(table_id)
        except NotFound:
            schema = [
                bigquery.SchemaField(model_col, bigquery.SqlTypeNames.BYTES),
                bigquery.SchemaField(data_col, bigquery.SqlTypeNames.BYTES),
                bigquery.SchemaField(ctlr_col, bigquery.SqlTypeNames.BYTES),
            ]
            table_obj = bigquery.Table(table_id, schema=schema)
            client.create_table(table_obj)

        row = {
            model_col: model_bytes,
            data_col: history_bytes,
            ctlr_col: ctrl_bytes,
        }

        try:
            errors = client.insert_rows_json(table_id, [row])
            if errors:
                print("BigQuery upsert reported errors:", errors)
            else:
                print("BigQuery upsert successful to", table_id)
        except Exception as e:
            print("BigQuery upsert failed:", e)

        print("DATA DISTRIBUTED")
        return result

if __name__ == "__main__":
    Guard().main()

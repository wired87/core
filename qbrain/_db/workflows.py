from __future__ import annotations

import duckdb
from typing import Optional, Any
import os
import json
from typing import List
from pathlib import Path

try:
    import pandas as pd  # type: ignore
except Exception:  
    pd = None  # type: ignore

# ---------- CORE ----------

_DEFAULT_DUCK_PATH = str(Path(__file__).resolve().parent.parent / "local.duckdb")


def db_connect(path: str = _DEFAULT_DUCK_PATH):
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    try:
        return duckdb.connect(path)
    except Exception as e:
        # DuckDB uses OS-level file locks for write mode.
        # When multiple local processes start (dev server, tests, shells), fall back
        # to a per-process DB file instead of crashing at import time.
        msg = str(e).lower()
        if "being used by another process" in msg or "file is already open" in msg:
            alt = str(Path(path).with_name(f"{Path(path).stem}.{os.getpid()}{Path(path).suffix}"))
            alt_dir = os.path.dirname(alt)
            if alt_dir and not os.path.exists(alt_dir):
                os.makedirs(alt_dir)
            print(f"[WARN] DuckDB file locked: {path}. Falling back to {alt}")
            return duckdb.connect(alt)
        raise


def db_close(con):
    con.close()


def duck_insert(con, table_name: str, rows: List[dict], upsert=False):
    if not isinstance(rows, list):
        rows = [rows]

    if not rows:
        print("No rows to process.")
        return True

    # ---------- 1. Flatten complex fields ----------
    cleaned_rows = []
    for row in rows:
        clean_row = {}
        for k, v in row.items():
            if isinstance(v, (list, dict)):
                clean_row[k] = json.dumps(v)
            else:
                clean_row[k] = v
        cleaned_rows.append(clean_row)

    # ---------- 2. Ensure table exists ----------
    # Union of all keys across rows so rows with different columns work
    all_keys = set()
    for row in cleaned_rows:
        all_keys.update(row.keys())
    columns = sorted(all_keys)

    col_defs = []
    for col in columns:
        # Infer type from first non-None value; all cols optional (nullable)
        sample = None
        for row in cleaned_rows:
            if col in row and row[col] is not None:
                sample = row[col]
                break
        if isinstance(sample, int):
            dtype = "BIGINT"
        elif isinstance(sample, float):
            dtype = "DOUBLE"
        else:
            dtype = "VARCHAR"
        if upsert and col == "id":
            col_defs.append(f"{col} {dtype} PRIMARY KEY")
        else:
            col_defs.append(f"{col} {dtype}")

    schema = ', '.join(col_defs)
    db_create_table(con, table_name, schema)

    # ---------- 3. Batch Insert ----------
    batch_size = 50
    cols_str = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))

    for i in range(0, len(cleaned_rows), batch_size):
        batch_chunk = cleaned_rows[i:i + batch_size]

        values = [
            tuple(row.get(col, None) for col in columns)
            for row in batch_chunk
        ]

        if upsert:
            if "id" not in columns:
                raise ValueError("duck_insert(..., upsert=True) requires an 'id' column")

            update_clause = ", ".join([f"{col}=excluded.{col}" for col in columns if col != "id"])
            try:
                con.executemany(
                    f"""
                    INSERT INTO {table_name} ({cols_str})
                    VALUES ({placeholders})
                    ON CONFLICT(id) DO UPDATE SET
                    {update_clause}
                    """,
                    values,
                )
            except Exception:
                # Fallback for legacy tables without a PK/unique constraint on id.
                # DuckDB requires a constraint for ON CONFLICT; emulate upsert via delete+insert.
                id_idx = columns.index("id")
                con.executemany(
                    f"DELETE FROM {table_name} WHERE id = ?",
                    [(v[id_idx],) for v in values],
                )
                con.executemany(
                    f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders})",
                    values,
                )
        else:
            con.executemany(f"INSERT INTO {table_name} ({cols_str}) VALUES ({placeholders})", values)

    return True




# ---------- MODULE-LEVEL FUNCTIONS (con passed explicitly) ----------

def db_exec(con, sql: str, params: Optional[List[Any]] = None):
    if params:
        return con.execute(sql, params)
    return con.execute(sql)


def db_query(con, sql: str, params: Optional[List[Any]] = None):
    if params:
        return con.execute(sql, params).fetchall()
    return con.execute(sql).fetchall()


def db_query_df(con, sql: str):
    if pd is None:
        raise ImportError("pandas is required for db_query_df(). Install pandas to use this helper.")
    return con.execute(sql).df()


# ---------- TABLE MANAGEMENT ----------

def db_create_table(con, table_name: str, schema_sql: str):
    con.execute(f"CREATE TABLE IF NOT EXISTS {table_name} ({schema_sql})")


def db_drop_table(con, table_name: str):
    con.execute(f"DROP TABLE IF EXISTS {table_name}")


# ---------- INSERT / UPDATE / DELETE ----------

def db_insert(con, table: str, columns: List[str], values: List[Any]):
    placeholders = ",".join(["?"] * len(values))
    cols = ",".join(columns)
    con.execute(
        f"INSERT INTO {table} ({cols}) VALUES ({placeholders})",
        values
    )


def db_update(con, table: str, set_clause: str, where_clause: str):
    con.execute(f"UPDATE {table} SET {set_clause} WHERE {where_clause}")


def db_delete(con, table: str, where_clause: str):
    con.execute(f"DELETE FROM {table} WHERE {where_clause}")


# ---------- DATAFRAME SUPPORT ----------

def db_register_df(con, df: pd.DataFrame, view_name: str):
    if pd is None:
        raise ImportError("pandas is required for db_register_df(). Install pandas to use this helper.")
    con.register(view_name, df)


def db_insert_df(con, table_name: str, df: pd.DataFrame):
    if pd is None:
        raise ImportError("pandas is required for db_insert_df(). Install pandas to use this helper.")
    con.register("tmp_df", df)
    con.execute(f"INSERT INTO {table_name} SELECT * FROM tmp_df")


# ---------- FILE IMPORT ----------

def db_read_csv(con, path: str, table_name: str):
    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{path}')"
    )


def db_read_parquet(con, path: str, table_name: str):
    con.execute(
        f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_parquet('{path}')"
    )


# ---------- FILE EXPORT ----------

def db_export_csv(con, table_name: str, path: str):
    con.execute(
        f"COPY {table_name} TO '{path}' (HEADER, DELIMITER ',')"
    )


def db_export_parquet(con, table_name: str, path: str):
    con.execute(
        f"COPY {table_name} TO '{path}' (FORMAT PARQUET)"
    )
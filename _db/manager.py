"""
DB Manager: unified interface for DuckDB (local) and BigQuery (remote).
"""
import os
import re
from typing import Any, Dict, List, Optional, Union


from _db.workflows import (
    db_connect,
    db_close,
    db_exec,
    db_create_table,
    duck_insert,
)
_db_mgr: Optional["DBManager"] = None
LOCAL_DB:str = os.getenv("LOCAL_DB", "True")

def get_db_manager() -> "DBManager":
    """Return QBrainTableManager. Uses BigQuery when LOCAL_DB=False and bqcore given; else DuckDB."""
    global _db_mgr

    if _db_mgr is None:
        _db_mgr = DBManager(
            local=LOCAL_DB == "True",
            dataset_id=os.getenv("PROJECT"),
        )
    return _db_mgr

class DBManager:
    """
    Unified DB manager: DuckDB when local=True, BigQuery when local=False.
    """

    def __init__(
        self,
        local: bool = True,
        duck_path: str = "local.duckdb",
        dataset_id: Optional[str] = None,
    ):
        """
        Args:
            local: If True, use DuckDB; if False, use BigQuery.
            duck_path: Path for DuckDB file (used when local=True).
            bqcore: BQCore instance for BigQuery (required when local=False).
            dataset_id: Dataset ID for BQ; used to create BQCore if bqcore not provided.
        """
        self.local = local

        if local:
            self._con = db_connect(duck_path)
            self._bqcore = None
            self.pid = os.getenv("PROJECT", "local")
            self.ds_id = os.getenv("PROJECT", "local")
            self.ds_ref = os.getenv("PROJECT", "local")
            self.get_table_schema = self._duck_get_table_schema
        else:
            self._con = None
            if dataset_id:
                from a_b_c.bq_agent._bq_core.bq_handler import BQCore
                bqcore = BQCore(dataset_id=dataset_id)

                self._bqcore = bqcore
                self.pid = bqcore.pid
                self.bqclient = bqcore.bqclient
                self.ds_id = bqcore.ds_id
                self.ds_ref = bqcore.ds_ref or f"{bqcore.pid}.{bqcore.ds_id}"
                self.get_table_schema = bqcore.get_table_schema

    def close(self):
        """Close connection (DuckDB only)."""
        if self._con:
            db_close(self._con)
            self._con = None

    def run_query(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None,
        conv_to_dict: bool = False,
        job_config=None,
    ) -> Union[List[Dict], List, None]:
        """
        Execute SELECT query and return results.

        Args:
            sql: SQL string. Use @param for BigQuery; @param converted to ? for DuckDB.
            params: Dict of param name -> value.
            conv_to_dict: Return list of dicts instead of rows.
            job_config: BigQuery QueryJobConfig (used when local=False, overrides params).
        """
        if self.local:
            if params is None and job_config and hasattr(job_config, "query_parameters") and job_config.query_parameters:
                params = {p.name: p.value for p in job_config.query_parameters}
            return self._run_query_duck(sql, params, conv_to_dict)
        return self._run_query_bq(sql, params, conv_to_dict, job_config)


    def _run_duck(self, query, conv_to_dict=False, job_config=None, bind_values=None):
        """Execute query on DuckDB."""
        import re
        if bind_values is None and job_config and hasattr(job_config, "query_parameters") and job_config.query_parameters:
            bind_values = {p.name: p.value for p in job_config.query_parameters}
        if bind_values:
            if isinstance(bind_values, dict):
                ordered = []
                for m in re.finditer(r"@(\w+)", query):
                    k = m.group(1)
                    if k in bind_values:
                        ordered.append(bind_values[k])
                query = re.sub(r"@\w+", "?", query)
            else:
                ordered = list(bind_values)
            cur = self._con.execute(query, ordered)
        else:
            cur = self._con.execute(query)
        result = cur.fetchall()
        if conv_to_dict and result:
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in result]
        return list(result) if result else []


    def run_db(self, query, conv_to_dict=False, job_config=None, params=None):
        """
        Single entry point for DB operations. Uses BigQuery or DuckDB based on init.
        Accepts params dict (for both) or job_config (BQ only).
        """
        if self.bqcore is not None:
            if job_config is None and params:
                from google.cloud import bigquery
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[bigquery.ScalarQueryParameter(k, "STRING", str(v)) for k, v in params.items()]
                )
            return self.bqcore.run_query(query, conv_to_dict=conv_to_dict, job_config=job_config)
        return self._run_duck(query, conv_to_dict=conv_to_dict, job_config=job_config, bind_values=params)



    def _run_query_duck(
        self,
        sql: str,
        bind_values: Optional[Union[Dict[str, Any], List[Any]]],
        conv_to_dict: bool,
    ):
        """Execute query on DuckDB. bind_values: dict for @param, list for ? placeholders."""
        if bind_values:
            if isinstance(bind_values, dict):
                # Convert @param to ? in left-to-right order; build ordered values
                ordered = []
                for m in re.finditer(r"@(\w+)", sql):
                    k = m.group(1)
                    if k in bind_values:
                        ordered.append(bind_values[k])
                sql = re.sub(r"@\w+", "?", sql)
                cur = self._con.execute(sql, ordered)
            else:
                # bind_values is a list (positional) for queries with ? placeholders
                cur = self._con.execute(sql, list(bind_values))
        else:
            cur = self._con.execute(sql)
        result = cur.fetchall()
        if conv_to_dict and result:
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in result]
        return list(result) if result else []

    def _run_query_bq(
        self,
        sql: str,
        params: Optional[Dict[str, Any]],
        conv_to_dict: bool,
        job_config,
    ):
        """Execute query on BigQuery."""
        if job_config is None and params:
            from google.cloud import bigquery
            job_config = bigquery.QueryJobConfig(
                query_parameters=[
                    bigquery.ScalarQueryParameter(k, "STRING", str(v))
                    for k, v in params.items()
                ]
            )
        return self._bqcore.run_query(
            sql, conv_to_dict=conv_to_dict, job_config=job_config
        )

    def execute(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ):
        """
        Execute DDL/DML (INSERT, UPDATE, DELETE, CREATE, etc.).
        No result returned.
        """
        if self.local:
            if isinstance(params, dict):
                ordered = []
                for m in re.finditer(r"@(\w+)", sql):
                    k = m.group(1)
                    if k in params:
                        ordered.append(params[k])
                sql = re.sub(r"@\w+", "?", sql)
                db_exec(self._con, sql, ordered)
            else:
                db_exec(self._con, sql, params)
        else:
            job_config = None
            if isinstance(params, dict):
                from google.cloud import bigquery
                job_config = bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter(k, "STRING", str(v))
                        for k, v in params.items()
                    ]
                )
            self._bqcore.run_query(sql, job_config=job_config)

    def insert(
        self,
        table: str,
        rows: Union[Dict, List[Dict]],
        upsert: bool = False,
    ) -> bool:
        """
        Insert rows into table.
        Uses DuckDB (duck_insert) when local, BQ insert when remote.
        """
        if not isinstance(rows, list):
            rows = [rows]
        if not rows:
            return True

        if self.local:
            return duck_insert(self._con, table, rows, upsert=upsert)
        # BigQuery
        return self._bqcore.bq_insert(table, rows, upsert=upsert)

    def create_table(self, table_name: str, schema_sql: str):
        """Create table if not exists (DuckDB only; BQ uses schema management)."""
        if self.local:
            db_create_table(self._con, table_name, schema_sql)
        else:
            raise NotImplementedError(
                "create_table for BigQuery: use bqcore.get_table_schema / ensure_table_exists"
            )

    def insert_col(self, table_id: str, column_name: str, column_type: str):
        """
        Add column to table if it does not exist.
        BQ: delegates to bqcore.insert_col
        DuckDB: ALTER TABLE ADD COLUMN (checks first, DuckDB has no IF NOT EXISTS).
        """
        if not self.local:
            return self._bqcore.insert_col(table_id, column_name, column_type)
        return self._duck_insert_col(table_id, column_name, column_type)

    def _duck_insert_col(self, table_id: str, column_name: str, column_type: str):
        """DuckDB: add column if not exists (check information_schema first)."""
        try:
            r = self._con.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ? AND column_name = ?",
                [table_id, column_name],
            ).fetchone()
            if r:
                return
            col_type = self._bq_to_duck_type(column_type)
            self._con.execute(f"ALTER TABLE {table_id} ADD COLUMN {column_name} {col_type}")
        except Exception as e:
            print(f"[WARN] DuckDB add column: {e}")

    def _bq_to_duck_type(self, bq_type: str) -> str:
        """Map BigQuery column types to DuckDB."""
        t = bq_type.upper().strip()
        if t.startswith("ARRAY<") and t.endswith(">"):
            inner = t[6:-1].strip()  # e.g. FLOAT64
            inner_duck = self._bq_to_duck_type(inner)
            return f"{inner_duck}[]"
        m = {
            "STRING": "VARCHAR",
            "INT64": "BIGINT",
            "INTEGER": "BIGINT",
            "FLOAT64": "DOUBLE",
            "BOOL": "BOOLEAN",
            "TIMESTAMP": "TIMESTAMP",
            "JSON": "VARCHAR",  # DuckDB JSON stored as text
        }
        return m.get(t, bq_type)

    def _duck_get_table_schema(
        self, table_id: str, schema: Dict[str, str], create_if_not_exists: bool = True
    ) -> Dict[str, str]:
        """DuckDB: ensure table exists with schema, return current schema."""
        r = self._con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema = 'main' AND table_name = ?",
            [table_id],
        ).fetchone()
        if r:
            rows = self._con.execute(
                "SELECT column_name, data_type FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ?",
                [table_id],
            ).fetchall()
            return {row[0]: row[1] for row in rows}
        if create_if_not_exists:
            col_defs = [f"{col} {self._bq_to_duck_type(dt)}" for col, dt in schema.items()]
            db_create_table(self._con, table_id, ", ".join(col_defs))
        return schema

    def showup(
        self,
        table_name: Optional[str] = None,
        limit: int = 100,
    ) -> None:
        """
        Render table data in console using rich Table.
        Handles both DuckDB and BigQuery.

        Args:
            table_name: Table to display. If None, lists all tables and shows data from each.
            limit: Max rows per table to display.
        """
        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            print("[WARN] rich not installed. Run: pip install rich")
            return

        console = Console()

        def _table_ref(t: str) -> str:
            if self.local:
                return t
            return f"`{self.pid}.{self.ds_id}.{t}`"

        def _get_tables() -> List[str]:
            if self.local:
                rows = self._con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
                ).fetchall()
                return [r[0] for r in rows]
            if self._bqcore is None:
                return []
            tables = self._bqcore.list_tables()
            return tables or []

        def _render_table(tbl_name: str) -> None:
            query = f"SELECT * FROM {_table_ref(tbl_name)} LIMIT {limit}"
            try:
                rows = self.run_query(query, conv_to_dict=True)
            except Exception as e:
                console.print(f"[red]Error querying {tbl_name}: {e}[/red]")
                return
            if not rows:
                console.print(f"[dim]{tbl_name}: (empty)[/dim]")
                return
            table = Table(title=tbl_name, show_header=True, header_style="bold cyan")
            for col in rows[0].keys():
                table.add_column(col, overflow="fold", max_width=40)
            for row in rows:
                table.add_row(*[str(row.get(c, ""))[:80] for c in rows[0].keys()])
            console.print(table)
            if len(rows) >= limit:
                console.print(f"[dim]... (limited to {limit} rows)[/dim]\n")

        if table_name:
            _render_table(table_name)
        else:
            tables = _get_tables()
            if not tables:
                console.print("[yellow]No tables found.[/yellow]")
                return
            for t in tables:
                _render_table(t)
                console.print()

    @property
    def connection(self):
        """DuckDB connection (when local) or None."""
        return self._con

    @property
    def bqcore(self):
        """BQCore instance (when not local) or None."""
        return self._bqcore

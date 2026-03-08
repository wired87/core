"""
DB Manager: unified interface for DuckDB (local).
"""
import json
import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from duckdb import DuckDBPyConnection

from qbrain._db.log_facade import db_log
from qbrain._db.workflows import (
    db_check,
    db_connect,
    db_close,
    db_exec,
    db_create_table,
    db_status,
    duck_insert,
)

_db_mgr: Optional["DBManager"] = None


def get_db_manager() -> "DBManager":
    global _db_mgr
    if _db_mgr is None:
        _db_mgr = DBManager()
    return _db_mgr


class DBManager:
    """
    DuckDB manager.
    """

    def __init__(self):
        self._con:DuckDBPyConnection = db_connect()

    def close(self):
        if self._con:
            db_close(self._con)
            self._con = None

    def create_sql_schema(self, schema):
        cols = []
        for k, v in schema.items():
            cols.append(f"{k} {v}")
        schema_sql = ", ".join(cols)
        return schema_sql


    def run_query(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
        conv_to_dict: bool = False,
    ):
        if params:
            if isinstance(params, dict):
                ordered = []
                for m in re.finditer(r"@(\w+)", sql):
                    k = m.group(1)
                    if k in params:
                        ordered.append(params[k])
                sql = re.sub(r"@\w+", "?", sql)
                cur = self._con.execute(sql, ordered)
            else:
                cur = self._con.execute(sql, list(params))
        else:
            cur = self._con.execute(sql)

        result = cur.fetchall()
        print("result:", len(result))

        if conv_to_dict and result:
            cols = [d[0] for d in cur.description] if cur.description else []
            return [dict(zip(cols, row)) for row in result]

        return list(result) if result else []


    def execute(
        self,
        sql: str,
        params: Optional[Union[Dict[str, Any], List[Any]]] = None,
    ):
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


    def insert(
        self,
        table: str,
        rows: Union[Dict, List[Dict]],
        upsert: bool = False,
    ) -> bool:
        db_log("info", "insert", table=table, rows=len(rows), upsert=upsert)
        if not isinstance(rows, list):
            rows = [rows]

        if not rows:
            return True

        schema = self._duck_get_table_schema(table, create_if_not_exists=False)

        new_rows = []

        for row in rows:
            new_row = {}

            for col, val in row.items():
                if not isinstance(val, (str, datetime)):
                    val = json.dumps(val)

                new_row[col] = val

                if col not in schema:
                    self._duck_insert_col(
                        table,
                        col,
                    )
            new_rows.append(new_row)

        ok = duck_insert(self._con, table, new_rows, upsert=upsert)
        if not ok:
            db_log("error", "insert failed", table=table)
        return ok

    def del_entry(self, nid: str, table: str, user_id: str, name_id: str = "id") -> bool:
        """
        Hard delete entry from DuckDB table.
        """
        try:
            sql = f"""
            DELETE FROM {table}
            WHERE {name_id} = ? AND user_id = ?
            """
            self._con.execute(sql, [nid, user_id])
            return True
        except Exception as e:
            return False

    def create_table(self, table_name: str, schema_sql: str):
        db_create_table(
            self._con,
            table_name,
            schema_sql,
        )

    def insert_col(self, table_id: str, column_name: str, column_type: str):
        return self._duck_insert_col(table_id, column_name, column_type)

    def _duck_insert_col(self, table_id: str, column_name: str):
        try:
            r = self._con.execute(
                "SELECT 1 FROM information_schema.columns WHERE table_schema = 'main' AND table_name = ? AND column_name = ?",
                [table_id, column_name],
            ).fetchone()

            if r:
                return

            col_type = "STRING"
            self._con.execute(
                f"ALTER TABLE {table_id} ADD COLUMN {column_name} {col_type}"
            )

        except Exception as e:
            db_log("error", "Err _duck_insert_col", error=str(e), table=table_id)



    def _duck_get_table_schema(
        self,
        table_id: str,
        create_if_not_exists: bool = True,
        schema: Optional[Dict[str, str]] = None,
    ) -> Dict[str, str]:

        schema = schema or {}

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

        if create_if_not_exists and schema:
            col_defs = [
                f"{col} STRING"
                for col, dt in schema.items()
            ]
            db_create_table(self._con, table_id, ", ".join(col_defs))
            return schema
        return schema

    def showup(
        self,
        table_name: Optional[str] = None,
        limit: int = 100,
    ) -> None:

        try:
            from rich.console import Console
            from rich.table import Table
        except ImportError:
            db_log("warn", "rich not installed. Run: pip install rich")
            return

        console = Console()

        def _get_tables() -> List[str]:
            rows = self._con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main' ORDER BY table_name"
            ).fetchall()
            return [r[0] for r in rows]

        def _render_table(tbl_name: str):

            query = f"SELECT * FROM {tbl_name} LIMIT {limit}"

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

    def print_table(self, table_name: str, limit: Optional[int] = 1000) -> None:
        """
        Render all items of a specific table in the rich terminal.
        Uses DuckDB .show() when possible; falls back to showup() on encoding errors.
        """
        limit_clause = f" LIMIT {limit}" if limit else ""
        query = f"SELECT * FROM {table_name}{limit_clause}"
        try:
            self._con.sql(query).show()
        except (UnicodeEncodeError, UnicodeDecodeError):
            self.showup(table_name=table_name, limit=limit or 1000)
        except Exception as e:
            db_log("error", "print_table failed", table=table_name, error=str(e))

    def status(self) -> dict:
        """Return DB status: path, tables, connection_alive."""
        return db_status(self._con)

    def check(self) -> bool:
        """Quick health check: connection alive and DB reachable."""
        return db_check(self._con)

    def get_state(self) -> dict:
        """Extended state: status + table row counts (for debug)."""
        from qbrain._db.config import duck_db_verbose

        state = db_status(self._con)
        if duck_db_verbose() >= 2:
            counts = {}
            for tbl in state.get("tables", []):
                try:
                    r = self._con.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()
                    counts[tbl] = r[0] if r else 0
                except Exception:
                    counts[tbl] = None
            state["row_counts"] = counts
        return state

    @property
    def connection(self):
        return self._con


if __name__ == "__main__":
    db = DBManager()
    db.print_table("params")       # Shows up to 1000 rows
    #db.print_table("params", None) # Shows all rows

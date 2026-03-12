"""
MIT RAY UND ANDERN TOOLS KANN ICH KOMPLETT GCP NACHABUEN UND FAHRE SO VIEL GÜNSTIGER

pip install duckdb
- seamless itegration with bigquery sql

"""

from qbrain._db import queries as queries
from qbrain._db.log_facade import db_log
from qbrain._db.manager import DBManager, get_db_manager
from qbrain._db.manager import db_check, db_status


__all__ = [
    "queries",
    "DBManager",
    "get_db_manager",
    "db_status",
    "db_check",
    "db_log",
]

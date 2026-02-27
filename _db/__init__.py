"""
MIT RAY UND ANDERN TOOLS KANN ICH KOMPLETT GCP NACHABUEN UND FAHRE SO VIEL GÜNSTIGER

pip install duckdb
- seamless itegration with bigquery sql

"""

from _db import queries as queries
from _db.manager import DBManager
from _db.vector_store import VectorStore


__all__ = ["queries", "DBManager", "VectorStore"]
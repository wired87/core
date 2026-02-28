"""
MIT RAY UND ANDERN TOOLS KANN ICH KOMPLETT GCP NACHABUEN UND FAHRE SO VIEL GÜNSTIGER

pip install duckdb
- seamless itegration with bigquery sql

"""

from qbrain._db import queries as queries
from qbrain._db.manager import DBManager
from qbrain._db.vector_store import VectorStore


__all__ = ["queries", "DBManager", "VectorStore"]
"""
QBRAIN Table Manager

Centralized table management for the entire QBRAIN dataset.
Handles schema definitions and table creation for all QBRAIN tables.
"""

from typing import Optional
import os
import dotenv

from qbrain._db.manager import get_db_manager, DBManager
from google import genai

dotenv.load_dotenv()

_QBRAIN_DEBUG = "[QBrainTableManager]"


class QBrainTableManager:
    """
    Centralized manager for all QBRAIN dataset tables.
    Handles schema definitions and table creation/verification.
    Receives BQCore instance via constructor (no inheritance).
    """

    DATASET_ID = "QBRAIN"

    def __init__(self):
        self.db = db
        # Use plain table names for DuckDB (LOCAL_DB=True); prefixed for BigQuery
        self._local = os.getenv("LOCAL") or os.getenv("LOCAL_DB", "True") == "True"
        self.pid = os.getenv("PID") or "QBRAIN"
        self._table_ref = lambda t: t if self._local else f"{self.pid}.{self.DATASET_ID}.{t}"

        _gemini_key = os.environ.get("GEMINI_API_KEY")
        self.genai_client = genai.Client(api_key=_gemini_key) if _gemini_key else None
        print(f"{_QBRAIN_DEBUG} initialized with dataset: {self.DATASET_ID}")








# Default singleton for standalone use (no orchestrator context)

db: Optional["DBManager"] = None
LOCAL_DB:str = os.getenv("LOCAL_DB", "True")

def get_qbrain_table_manager(bqcore=None) -> "QBrainTableManager":
    """Return QBrainTableManager. Uses BigQuery when LOCAL_DB=False and bqcore given; else DuckDB."""
    global db
    if db is None:
        db = get_db_manager()
    return db

db
if __name__ == "__main__":
    qbrain_manager = get_qbrain_table_manager()
    qbrain_manager.reset_tables(["fields"])

"""
Production-ready vector store for semantic search, RAG, and similarity retrieval.

Backend: DuckDB (default). Uses FLOAT[] for vectors and list_cosine_similarity.
Designed for easy replacement with Qdrant/FAISS in future.

Concurrency: Safe for single-process multi-thread use (RLock guards all operations).
For multi-process: use separate db paths per process, or external coordination;
DuckDB supports single writer, multiple readers when using a shared file.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from qbrain._db.workflows import db_connect, db_close, db_create_table

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Schema
# -----------------------------------------------------------------------------

_VECTOR_TABLE_SCHEMA = """
id VARCHAR PRIMARY KEY,
embedding FLOAT[] NOT NULL,
metadata VARCHAR,
created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
"""

# -----------------------------------------------------------------------------
# VectorStore
# -----------------------------------------------------------------------------


class VectorStore:
    """
    High-performance vector storage and retrieval using DuckDB.

    Supports semantic search, RAG, similarity search (cosine/dot/L2),
    metadata filtering, batch ingestion, upserts, and deletions.
    """

    def __init__(
        self,
        store_name: str = "vectors",
        db_path: Optional[str] = None,
        *,
        dimension: Optional[int] = None,
        normalize_embeddings: bool = True,
        batch_size: int = 100,
    ):
        """
        Args:
            store_name: Logical name for the vector table (used as table name).
            db_path: Path to DuckDB file. Default: {cwd}/vector_store.duckdb.
            dimension: Expected vector dimension. If set, validates on add/upsert.
            normalize_embeddings: If True, L2-normalize vectors before storage (cosine = dot).
            batch_size: Default batch size for add_vectors / upsert_vectors.
        """
        self._store_name = self._sanitize_table_name(store_name)
        self._db_path = db_path or str(Path.cwd() / "vector_store.duckdb")
        self._dimension = dimension
        self._normalize = normalize_embeddings
        self._batch_size = batch_size
        self._con = None
        self._lock = threading.RLock()

    def _sanitize_table_name(self, name: str) -> str:
        """Ensure table name is safe for SQL."""
        return "".join(c if c.isalnum() or c == "_" else "_" for c in name) or "vectors"

    def _get_connection(self):
        """Lazy connection; thread-safe via lock in public API."""
        if self._con is None:
            raise RuntimeError("VectorStore is closed or never opened. Call create_store() first.")
        return self._con

    def _ensure_connection(self) -> None:
        """Open connection if closed."""
        if self._con is None:
            if self._db_path != ":memory:":
                directory = os.path.dirname(self._db_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory)
            self._con = db_connect(self._db_path)
            logger.debug("VectorStore connected to %s", self._db_path)

    def _to_list(self, v: Union[np.ndarray, Sequence[float]]) -> List[float]:
        """Convert to list of float32 for storage."""
        arr = np.asarray(v, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Vector must be 1-dimensional")
        if self._dimension is not None and arr.size != self._dimension:
            raise ValueError(
                f"Vector dimension {arr.size} does not match expected {self._dimension}"
            )
        if self._normalize:
            norm = np.linalg.norm(arr)
            if norm > 1e-12:
                arr = arr / norm
            else:
                arr = np.zeros_like(arr)
        return arr.tolist()

    def _metadata_to_str(self, metadata: Optional[Dict[str, Any]]) -> Optional[str]:
        """Serialize metadata to JSON string."""
        if metadata is None:
            return None
        return json.dumps(metadata, default=str)

    def _str_to_metadata(self, s: Optional[str]) -> Optional[Dict[str, Any]]:
        """Deserialize metadata from JSON string."""
        if s is None or s == "":
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return None

    def _build_filter_sql(self, filters: Optional[Dict[str, Any]]) -> tuple[str, list]:
        """Build WHERE clause and params for metadata filters. Uses json_extract."""
        if not filters:
            return "", []
        conditions = []
        params: List[Any] = []
        for i, (key, value) in enumerate(filters.items()):
            param_name = f"f{i}"
            conditions.append(f"json_extract_string(metadata, '$.{key}') = ?")
            params.append(str(value))
        return " AND " + " AND ".join(conditions), params

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def create_store(self) -> None:
        """
        Create the vector table if it does not exist.
        Opens the DB connection if not already open.
        Applies schema migrations if needed.
        """
        with self._lock:
            self._ensure_connection()
            con = self._get_connection()
            table = self._store_name
            db_create_table(con, table, _VECTOR_TABLE_SCHEMA.strip())
            self._migrate_schema(con, table)
            logger.info("VectorStore table %s ready", table)

    def _migrate_schema(self, con, table: str) -> None:
        """Apply schema migrations. Extend for future schema changes."""
        try:
            con.execute(
                "CREATE TABLE IF NOT EXISTS _vector_store_schema (store_name VARCHAR PRIMARY KEY, version INT)"
            )
            cur = con.execute(
                "SELECT version FROM _vector_store_schema WHERE store_name = ?",
                [table],
            )
            row = cur.fetchone()
            version = row[0] if row else 0
            if version < 1:
                con.execute(
                    """
                    INSERT INTO _vector_store_schema (store_name, version) VALUES (?, 1)
                    ON CONFLICT (store_name) DO UPDATE SET version = 1
                    """,
                    [table],
                )
        except Exception as e:
            logger.warning("Schema migration check skipped: %s", e)

    def add_vectors(
        self,
        ids: Sequence[str],
        vectors: Sequence[Union[np.ndarray, Sequence[float]]],
        metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        *,
        batch_size: Optional[int] = None,
    ) -> int:
        """
        Insert vectors. Fails if any id already exists.

        Args:
            ids: Unique identifiers for each vector.
            vectors: Embedding vectors (float32, 1D).
            metadata: Optional metadata dict per vector. None entries allowed.
            batch_size: Override default batch size.

        Returns:
            Number of vectors inserted.
        """
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have same length")
        meta = metadata if metadata is not None else [None] * len(ids)
        if len(meta) != len(ids):
            raise ValueError("metadata length must match ids")

        batch = batch_size or self._batch_size
        inserted = 0
        with self._lock:
            con = self._get_connection()
            table = self._store_name
            for i in range(0, len(ids), batch):
                chunk_ids = ids[i : i + batch]
                chunk_vecs = vectors[i : i + batch]
                chunk_meta = meta[i : i + batch]
                rows = [
                    (
                        str(pid),
                        self._to_list(v),
                        self._metadata_to_str(m),
                    )
                    for pid, v, m in zip(chunk_ids, chunk_vecs, chunk_meta)
                ]
                try:
                    con.execute("BEGIN TRANSACTION")
                    con.executemany(
                        f"INSERT INTO {table} (id, embedding, metadata) VALUES (?, ?, ?)",
                        rows,
                    )
                    con.execute("COMMIT")
                except Exception:
                    con.execute("ROLLBACK")
                    raise
                inserted += len(rows)
        logger.debug("add_vectors: inserted %d rows", inserted)
        return inserted

    def upsert_vectors(
        self,
        ids: Sequence[str],
        vectors: Sequence[Union[np.ndarray, Sequence[float]]],
        metadata: Optional[Sequence[Optional[Dict[str, Any]]]] = None,
        *,
        batch_size: Optional[int] = None,
    ) -> int:
        """
        Insert or replace vectors. Existing ids are updated.

        Returns:
            Number of vectors upserted.
        """
        if len(ids) != len(vectors):
            raise ValueError("ids and vectors must have same length")
        meta = metadata if metadata is not None else [None] * len(ids)
        if len(meta) != len(ids):
            raise ValueError("metadata length must match ids")

        batch = batch_size or self._batch_size
        upserted = 0
        with self._lock:
            con = self._get_connection()
            table = self._store_name
            for i in range(0, len(ids), batch):
                chunk_ids = ids[i : i + batch]
                chunk_vecs = vectors[i : i + batch]
                chunk_meta = meta[i : i + batch]
                rows = [
                    (
                        str(pid),
                        self._to_list(v),
                        self._metadata_to_str(m),
                    )
                    for pid, v, m in zip(chunk_ids, chunk_vecs, chunk_meta)
                ]
                try:
                    con.execute("BEGIN TRANSACTION")
                    con.executemany(
                        f"""
                        INSERT INTO {table} (id, embedding, metadata)
                        VALUES (?, ?, ?)
                        ON CONFLICT (id) DO UPDATE SET
                            embedding = excluded.embedding,
                            metadata = excluded.metadata
                        """,
                        rows,
                    )
                    con.execute("COMMIT")
                except Exception:
                    con.execute("ROLLBACK")
                    raise
                upserted += len(rows)
        logger.debug("upsert_vectors: upserted %d rows", upserted)
        return upserted

    def delete(self, ids: Union[str, Sequence[str]]) -> int:
        """
        Delete vectors by id(s).

        Args:
            ids: Single id or sequence of ids.

        Returns:
            Number of rows deleted.
        """
        id_list = [ids] if isinstance(ids, str) else list(ids)
        if not id_list:
            return 0
        with self._lock:
            con = self._get_connection()
            table = self._store_name
            placeholders = ",".join(["?"] * len(id_list))
            cur = con.execute(
                f"DELETE FROM {table} WHERE id IN ({placeholders})",
                [str(i) for i in id_list],
            )
            return cur.rowcount or 0

    def similarity_search(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
    ) -> List[Dict[str, Any]]:
        """
        Find top_k most similar vectors.

        Args:
            query_vector: Query embedding.
            top_k: Number of results to return.
            filters: Metadata filters (key=value). All must match.
            metric: "cosine" (default), "dot", or "l2".

        Returns:
            List of dicts: id, embedding, metadata, score.
        """
        q = self._to_list(query_vector)
        where_sql, where_params = self._build_filter_sql(filters)
        params: List[Any] = [q]
        params.extend(where_params)

        with self._lock:
            con = self._get_connection()
            table = self._store_name

            if metric == "cosine":
                score_expr = "list_cosine_similarity(embedding, ?)"
            elif metric == "dot":
                score_expr = "list_dot_product(embedding, ?)"
            elif metric == "l2":
                score_expr = "-list_distance(embedding, ?)"
            else:
                raise ValueError(f"Unknown metric: {metric}")

            sql = f"""
                SELECT id, embedding, metadata, {score_expr} AS score
                FROM {table}
                WHERE 1=1 {where_sql}
                ORDER BY score DESC
                LIMIT ?
            """
            params.append(top_k)
            cur = con.execute(sql, params)
            rows = cur.fetchall()

        result = []
        for row in rows:
            result.append({
                "id": row[0],
                "embedding": row[1],
                "metadata": self._str_to_metadata(row[2]),
                "score": float(row[3]),
            })
        return result

    def batch_similarity_search(
        self,
        query_vectors: Sequence[Union[np.ndarray, Sequence[float]]],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        metric: str = "cosine",
    ) -> List[List[Dict[str, Any]]]:
        """
        Run similarity_search for each query vector.

        Returns:
            List of result lists, one per query.
        """
        return [
            self.similarity_search(q, top_k=top_k, filters=filters, metric=metric)
            for q in query_vectors
        ]

    def classify(
        self,
        query_vector: Union[np.ndarray, Sequence[float]],
        labels: Sequence[str],
        filters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Classify query by finding nearest labeled vector.
        Expects metadata to contain a 'label' field. If labels is non-empty,
        only vectors whose metadata.label is in labels are considered.

        Args:
            query_vector: Query embedding.
            labels: Valid label values. If empty, any vector may match.
            filters: Additional metadata filters.

        Returns:
            The label of the nearest matching vector, or empty string if none.
        """
        combined = dict(filters or {})
        if labels:
            # Filter to vectors with label in labels (OR across labels)
            results = []
            for label in labels:
                combined["label"] = label
                r = self.similarity_search(query_vector, top_k=1, filters=combined)
                if r:
                    results.append((r[0]["score"], r[0].get("metadata", {}).get("label", "")))
            if not results:
                return ""
            results.sort(key=lambda x: x[0], reverse=True)
            return str(results[0][1])
        results = self.similarity_search(
            query_vector, top_k=1, filters=combined if combined else None
        )
        if not results:
            return ""
        return str(results[0].get("metadata", {}).get("label", ""))

    def count(self) -> int:
        """Return total number of vectors in the store."""
        with self._lock:
            con = self._get_connection()
            cur = con.execute(f"SELECT COUNT(*) FROM {self._store_name}")
            return int(cur.fetchone()[0])

    def reset(self) -> None:
        """Drop the vector table and recreate it. All data is lost."""
        with self._lock:
            con = self._get_connection()
            con.execute(f"DROP TABLE IF EXISTS {self._store_name}")
            db_create_table(con, self._store_name, _VECTOR_TABLE_SCHEMA.strip())
            logger.info("VectorStore %s reset", self._store_name)

    def optimize(self) -> None:
        """
        Run CHECKPOINT and VACUUM to optimize storage.
        Use after bulk deletes or large ingestions.
        """
        with self._lock:
            con = self._get_connection()
            con.execute("CHECKPOINT")
            con.execute("VACUUM")
            logger.debug("VectorStore optimized")

    def close(self) -> None:
        """Close the database connection. Safe to call multiple times."""
        with self._lock:
            if self._con is not None:
                db_close(self._con)
                self._con = None
                logger.debug("VectorStore closed")

    def __enter__(self) -> "VectorStore":
        self._ensure_connection()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# -----------------------------------------------------------------------------
# Usage example
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Create store (uses in-memory or temp file for demo)
    store = VectorStore(
        store_name="demo_vectors",
        db_path=":memory:",  # or "demo_vectors.duckdb"
        dimension=4,
        normalize_embeddings=True,
    )
    store.create_store()

    # Add vectors
    ids = ["a", "b", "c"]
    vectors = [
        [1.0, 0.0, 0.0, 0.0],
        [0.9, 0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]
    metadata = [{"label": "x", "source": "doc1"}, {"label": "x", "source": "doc2"}, {"label": "y", "source": "doc3"}]
    store.add_vectors(ids, vectors, metadata=metadata)

    # Similarity search
    query = [1.0, 0.05, 0.0, 0.0]
    results = store.similarity_search(query, top_k=2)
    print("Top 2:", [(r["id"], r["score"]) for r in results])

    # Filtered search
    results = store.similarity_search(query, top_k=2, filters={"label": "x"})
    print("Filtered (label=x):", [(r["id"], r["score"]) for r in results])

    # Count and close
    print("Count:", store.count())
    store.close()

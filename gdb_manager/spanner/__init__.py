

import dotenv


dotenv.load_dotenv()


from google.cloud import spanner

"""
PIPESyour-instance-id
1. 
CSV Files on Cloud Storage to BigQuery
"""

SP_INSTANCE_ID = "brainmaster"
SP_DATABASE_ID = "brainmaster01"
GRAPH_NAME = SP_INSTANCE_ID.upper()



ALL_INFO_SCHEMAS = [
    "USERS",
    "CHANGE_STREAM_COLUMNS",
    "CHANGE_STREAM_OPTIONS",
    "CHANGE_STREAM_PRIVILEGES",
    "CHANGE_STREAM_TABLES",
    "CHANGE_STREAMS",
    "CHECK_CONSTRAINTS",
    "COLUMN_COLUMN_USAGE",
    "COLUMN_OPTIONS",
    "COLUMN_PRIVILEGES",
    "COLUMNS",
    "CONSTRAINT_COLUMN_USAGE",
    "CONSTRAINT_TABLE_USAGE",
    "DATABASE_OPTIONS",
    "INDEX_COLUMNS",
    "INDEX_OPTIONS",
    "INDEXES",
    "KEY_COLUMN_USAGE",
    "MODEL_COLUMN_OPTIONS",
    "MODEL_COLUMNS",
    "MODEL_OPTIONS",
    "MODEL_PRIVILEGES",
    "MODELS",
    "PARAMETERS",
    "PLACEMENT_OPTIONS",
    "PLACEMENTS",
    "PROPERTY_GRAPHS",
    "REFERENTIAL_CONSTRAINTS",
    "ROLE_CHANGE_STREAM_GRANTS",
    "ROLE_COLUMN_GRANTS",
    "ROLE_GRANTEES",
    "ROLE_MODEL_GRANTS",
    "ROLE_ROUTINE_GRANTS",
    "ROLE_TABLE_GRANTS",
    "ROLES",
    "ROUTINE_OPTIONS",
    "ROUTINE_PRIVILEGES",
    "ROUTINES",
    "SCHEMATA",
    "SEQUENCE_OPTIONS",
    "SEQUENCES",
    "SPANNER_STATISTICS",
    "TABLE_CONSTRAINTS",
    "TABLE_PRIVILEGES",
    "TABLE_SYNONYMS",
    "TABLES",
    "VIEWS",
    "ACTIVE_PARTITIONED_DMLS",
    "ACTIVE_QUERIES_SUMMARY",
    "LOCK_STATS_TOP_10MINUTE",
    "LOCK_STATS_TOP_HOUR",
    "LOCK_STATS_TOP_MINUTE",
    "LOCK_STATS_TOTAL_10MINUTE",
    "LOCK_STATS_TOTAL_HOUR",
    "LOCK_STATS_TOTAL_MINUTE",
    "OLDEST_ACTIVE_QUERIES",
    "QUERY_PROFILES_TOP_10MINUTE",
    "QUERY_PROFILES_TOP_HOUR",
    "QUERY_PROFILES_TOP_MINUTE",
    "QUERY_RECOMMENDATIONS",
    "QUERY_STATS_TOP_10MINUTE",
    "QUERY_STATS_TOP_HOUR",
    "QUERY_STATS_TOP_MINUTE",
    "QUERY_STATS_TOTAL_10MINUTE",
    "QUERY_STATS_TOTAL_HOUR",
    "QUERY_STATS_TOTAL_MINUTE",
    "READ_STATS_TOP_10MINUTE",
    "TABLE_SIZES_STATS_PER_LOCALITY_GROUP_1HOUR",
    "LOCALITY_GROUP_OPTIONS",
    "READ_STATS_TOP_HOUR",
    "READ_STATS_TOP_MINUTE",
    "READ_STATS_TOTAL_10MINUTE",
    "READ_STATS_TOTAL_HOUR",
    "READ_STATS_TOTAL_MINUTE",
    "ROW_DELETION_POLICIES",
    "SPLIT_HOTNESS_STATS_TOP_MINUTE",
    "SPLIT_STATS_TOP_MINUTE",
    "SUPPORTED_OPTIMIZER_VERSIONS",
    "TABLE_OPERATIONS_STATS_10MINUTE",
    "TABLE_OPERATIONS_STATS_HOUR",
    "TABLE_OPERATIONS_STATS_MINUTE",
    "TABLE_SIZES_STATS_1HOUR",
    "TASKS",
    "TXN_STATS_TOP_10MINUTE",
    "TXN_STATS_TOP_HOUR",
    "TXN_STATS_TOP_MINUTE",
    "TXN_STATS_TOTAL_10MINUTE",
    "TXN_STATS_TOTAL_HOUR",
    "TXN_STATS_TOTAL_MINUTE",
    "USER_SPLIT_POINTS"
]

def create_default_table_query(table_name):
    if table_name:
        table_name = table_name.upper().replace("-", "_")
        return f"""
        CREATE TABLE {table_name} (
        id STRING(MAX) NOT NULL, 
        ) PRIMARY KEY (id)
        """





###################### QUERIES
def create_default_node_table_query(table_name):
    if table_name:
        table_name = table_name.upper().replace("-", "_")
        return f"""
        CREATE TABLE {table_name} (
        id STRING(MAX), 
        type STRING(MAX),
        parent ARRAY<STRING(MAX)>,
        child ARRAY<STRING(MAX)>,
        ) PRIMARY KEY (id)
        """


def delete_table_query(name):
    return f"""
    DROP TABLE
    `{name}`;
    """



def create_goterm_rag_result_table_query():

    return f"""
    CREATE TABLE GOTERMRAGRESULTS (
    id STRING(MAX),  
    job_id STRING(MAX),
    term STRING(MAX),
    keywords ARRAY<STRING(MAX)>,
    go_terms ARRAY<STRING(MAX)>,
    term_limit INT64,
    genes ARRAY<STRING(MAX)>,
    ) PRIMARY KEY (id)
    """









def create_default_auth_table_query(table_name):
    if table_name:
        table_name = table_name.upper()
        return f"""
        CREATE TABLE {table_name} (
        id STRING(MAX) NOT NULL, 
        email STRING(MAX) NOT NULL,
        password STRING(MAX) NOT NULL,
        created_at STRING(MAX),
        last_login STRING(MAX),
        role STRING(MAX),
        status STRING(MAX),
        documentai_id STRING(MAX),
        documentai_processor_type STRING(MAX),
        ) PRIMARY KEY (id)
        """



def create_default_edge_table_query(table_name):
    print("Create table:", table_name)
    table_name = table_name.replace("-", "_")
    return f"""
    CREATE TABLE {table_name} (
    id STRING(MAX) NOT NULL,
     src STRING(MAX) NOT NULL,
     trgt STRING(MAX) NOT NULL,
     rel STRING(MAX),
     src_layer STRING(MAX),
     trgt_layer STRING(MAX),
     type STRING(MAX),
    ) PRIMARY KEY (src, trgt)
    """

def create_metadata_table_query():
    return f"""
    CREATE TABLE METADATA (
    id STRING(MAX) NOT NULL,-- Name of the table
    metadata STRING(MAX), -- Whether NULL values are allowed
    ) PRIMARY KEY (id)
    """

def del_col(table, col):
    return f"ALTER TABLE {table.upper()} DROP COLUMN {col};"

def get_col_names():
    return """
    SELECT COLUMN_NAME
    FROM INFORMATION_SCHEMA.COLUMNS
    WHERE TABLE_NAME = @table_name
    """


def get_graph_table_names_query():
    return f"""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE GRAPH_NAME = '{GRAPH_NAME}'
    """


def get_specific_g_info():
    return f"SELECT * FROM INFORMATION_SCHEMA.GRAPHS WHERE GRAPH_NAME = '{GRAPH_NAME}'"



edge_table_references=f"""
        SELECT
            ccu.COLUMN_NAME,
            ccu.TABLE_NAME AS REFERENCED_TABLE
        FROM INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc
        JOIN INFORMATION_SCHEMA.CONSTRAINT_COLUMN_USAGE ccu
        ON tc.CONSTRAINT_NAME = ccu.CONSTRAINT_NAME
        WHERE tc.TABLE_NAME = @table_name
        AND tc.CONSTRAINT_TYPE = 'FOREIGN KEY'
        """

def add_graph_table(table_type: "NODE" or "EDGE", table):
    return f"""
        ALTER PROPERTY GRAPH {GRAPH_NAME}
        ADD {table_type} TABLE {table};
        """

#SELECT edge_tables, node_tables FROM metadata WHERE id = '{graph_name}'


def create_graph_query(node_tables, edge_table_schema, graph_name=None):
    return f"""
    CREATE PROPERTY GRAPH {graph_name or GRAPH_NAME}
    NODE TABLES ({', '.join(node_tables)})
    EDGE TABLES ({edge_table_schema})
    """


def get_edge_table_schema(edge_item_name, trgt_re_table, src_ref_table):
    return f"""
    {edge_item_name}
        SOURCE KEY (src) REFERENCES {src_ref_table} (id)
        DESTINATION KEY (trgt) REFERENCES {trgt_re_table} (id),
    """


def get_graph_query(graph_name):
    return f"""
    SELECT TABLE_NAME FROM INFORMATION_SCHEMA.GRAPHS WHERE GRAPH_NAME = '{graph_name}'
    """



def ask_graph():
    return """
    SELECT productId, productName, productDescription, inventoryCount, 
        COSINE_DISTANCE(productDescriptionEmbedding, 
            (SELECT embeddings.values 
             FROM ML.PREDICT(
                 MODEL EmbeddingsModel, 
                 (SELECT "I'd like to buy a starter bike for my 3-year-old child" AS content)
             )
            )
        ) AS distance
    FROM products
    WHERE inventoryCount > 0
    ORDER BY distance
    LIMIT 5;
    """


def all_ids_table_query(table_name):
    return f"SELECT id FROM {table_name}"



import google.cloud.spanner
from google.cloud.spanner_v1 import param_types

def embed_and_store_rows(
    instance_id: str,
    database_id: str,
    table_name: str,
    embedding_model_name: str,
    id_column: str,
    embedding_column: str,
    columns_to_embed: list = None,  # Optional: List of columns to embed
    separator: str = " | ",  # Optional: Separator for concatenation
    chunk_size: int = 100,  # Process in chunks for efficiency
):
    """
    Dynamically embeds specified columns (or all) of a Spanner table and stores
    the embeddings in a specified embedding column.

    Args:
        instance_id: Spanner instance ID.
        database_id: Spanner database ID.
        table_name: Name of the table to process.
        embedding_model_name: Name of the embedding model in Spanner.
        id_column: Name of the primary key column (used for row identification).
        embedding_column: Name of the column to store the embeddings.
        columns_to_embed: List of column names to embed. If None, all columns are embedded.
        separator: Separator to use when concatenating column values.
        chunk_size: Number of rows to process in each chunk.
    """

    spanner_client = google.cloud.spanner.Client()
    instance = spanner_client.instance(instance_id)
    database = instance.database(database_id)

    with database.batch() as batch:
        # Get all column names from the table if columns_to_embed is not provided
        if columns_to_embed is None:
            with database.snapshot() as snapshot:
                results = snapshot.execute_sql(
                    f"SELECT column_name FROM INFORMATION_SCHEMA.COLUMNS WHERE table_name = '{table_name}'"
                )
                columns_to_embed = [row[0] for row in results]

        # Fetch IDs and rows in chunks
        with database.snapshot() as snapshot:
            results = snapshot.execute_sql(
                f"SELECT {id_column}, * FROM {table_name} WHERE {embedding_column} IS NULL"
            )

            id_chunk = []
            row_chunk = []
            for row in results:
                id_chunk.append(row[0])
                row_chunk.append(row)

                if len(id_chunk) >= chunk_size:
                    _embed_and_update_chunk(
                        database,
                        batch,
                        table_name,
                        embedding_model_name,
                        id_column,
                        embedding_column,
                        columns_to_embed,
                        separator,
                        id_chunk,
                        row_chunk,
                    )
                    id_chunk = []
                    row_chunk = []

            # Process any remaining rows
            if id_chunk:
                _embed_and_update_chunk(
                    database,
                    batch,
                    table_name,
                    embedding_model_name,
                    id_column,
                    embedding_column,
                    columns_to_embed,
                    separator,
                    id_chunk,
                    row_chunk,
                )

def _embed_and_update_chunk(
    database,
    batch,
    table_name,
    embedding_model_name,
    id_column,
    embedding_column,
    columns_to_embed,
    separator,
    id_chunk,
    row_chunk,
):
    """Helper function to process a chunk of rows."""
    with database.snapshot() as snapshot:
        rows_for_embedding = []
        for row in row_chunk:
            row_dict = dict(zip(snapshot.column_names, row))
            row_values = [str(row_dict[col]) for col in columns_to_embed]
            rows_for_embedding.append(separator.join(row_values))

        # Generate embeddings
        results = snapshot.execute_sql(
            f"SELECT embeddings.values FROM ML.PREDICT(MODEL @model, (SELECT @contents AS content))",
            params={"model": embedding_model_name, "contents": rows_for_embedding},
            param_types={"model": param_types.STRING, "contents": param_types.ARRAY(param_types.STRING)},
        )

        embeddings = [row[0] for row in results]

        # Update the table with embeddings
        for row_id, embedding in zip(id_chunk, embeddings):
            batch.update(
                table=table_name,
                columns=(id_column, embedding_column),
                values=(f"e{row_id}", embedding),
            )


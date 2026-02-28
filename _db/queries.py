"""
Centralized SQL query definitions with dynamic args.
All queries use parameterized placeholders (@param) for safety.
"""
from typing import Any, Dict, List, Optional, Tuple
from google.cloud import bigquery


# ---------- BigQuery Query Builders ----------

def bq_get_users_entries(
    ds_ref: str,
    table: str,
    user_id: str,
    select: str = "*",
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get user entries (latest per id)."""
    query = f"""
        SELECT {select}
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
            FROM `{ds_ref}.{table}`
            WHERE (user_id = @user_id OR user_id = 'public') AND (status != 'deleted' OR status IS NULL)
        )
        WHERE row_num = 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
    )
    return query, job_config


def bq_list_session_entries(
    ds_ref: str,
    table: str,
    user_id: str,
    session_id: str,
    select: str = "*",
    partition_key: str = "id",
) -> Tuple[str, bigquery.QueryJobConfig]:
    """List entries linked to session (latest per partition)."""
    query = f"""
        SELECT {select}, status
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY {partition_key} ORDER BY created_at DESC) as row_num
            FROM `{ds_ref}.{table}`
            WHERE user_id = @user_id AND session_id = @session_id
        )
        WHERE row_num = 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
        ]
    )
    return query, job_config


def bq_get_envs_linked_rows(
    ds_ref: str,
    table_name: str,
    env_id: str,
    linked_row_id: str,
    linked_row_id_name: str,
    user_id: str,
    select: str = "*",
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get linked rows for env."""
    query = f"""
        SELECT {select}
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
            FROM `{ds_ref}.{table_name}`
            WHERE env_id = @env_id AND {linked_row_id_name} = @{linked_row_id_name} AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
        )
        WHERE row_num = 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("env_id", "STRING", env_id),
            bigquery.ScalarQueryParameter(linked_row_id_name, "STRING", linked_row_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_get_modules_linked_rows(
    ds_ref: str,
    table_name: str,
    module_id: str,
    linked_row_id: str,
    linked_row_id_name: str,
    user_id: str,
    select: str = "*",
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get linked rows for module."""
    query = f"""
        SELECT {select}
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
            FROM `{ds_ref}.{table_name}`
            WHERE module_id = @module_id AND {linked_row_id_name} = @{linked_row_id_name} AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
        )
        WHERE row_num = 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
            bigquery.ScalarQueryParameter(linked_row_id_name, "STRING", linked_row_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_row_from_id(
    ds_ref: str,
    table: str,
    ids: List[str],
    select: str = "*",
    user_id: Optional[str] = None,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get rows by id list (latest per id)."""
    query = f"""
        SELECT {select}
        FROM (
            SELECT {select}, ROW_NUMBER() OVER (PARTITION BY id ORDER BY created_at DESC) as row_num
            FROM `{ds_ref}.{table}`
            WHERE id IN UNNEST(@id) AND (status != 'deleted' OR status IS NULL)
        )
        WHERE row_num = 1
    """
    params = [bigquery.ArrayQueryParameter("id", "STRING", ids)]
    if user_id:
        query += " AND user_id = @user_id"
        params.append(bigquery.ScalarQueryParameter("user_id", "STRING", user_id))
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return query, job_config


def bq_upsert_copy_select(
    table_ref: str,
    keys: Dict[str, Any],
) -> Tuple[str, bigquery.QueryJobConfig]:
    """SELECT latest row for upsert_copy (fetch before insert)."""
    where_clause = " AND ".join([f"{k} = @{k}" for k in keys.keys()])
    query = f"""
        SELECT * FROM `{table_ref}`
        WHERE {where_clause}
        ORDER BY created_at DESC LIMIT 1
    """
    params = []
    for k, v in keys.items():
        if isinstance(v, int):
            params.append(bigquery.ScalarQueryParameter(k, "INT64", int(v)))
        else:
            params.append(bigquery.ScalarQueryParameter(k, "STRING", str(v)))
    job_config = bigquery.QueryJobConfig(query_parameters=params)
    return query, job_config


def bq_alter_add_column(
    table_ref: str,
    col_name: str,
    col_type: str,
) -> str:
    """ALTER TABLE ADD COLUMN."""
    return f"ALTER TABLE `{table_ref}` ADD COLUMN IF NOT EXISTS {col_name} {col_type}"


def bq_update_sessions_methods_soft_delete(
    pid: str,
    dataset_id: str,
    table: str,
    method_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Soft delete sessions_to_methods links."""
    query = f"""
        UPDATE `{pid}.{dataset_id}.{table}`
        SET status = 'deleted'
        WHERE method_id = @method_id AND user_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("method_id", "STRING", method_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_update_sessions_modules_soft_delete(
    pid: str,
    dataset_id: str,
    table: str,
    module_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Soft delete sessions_to_modules links."""
    query = f"""
        UPDATE `{pid}.{dataset_id}.{table}`
        SET status = 'deleted'
        WHERE module_id = @module_id AND user_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_update_modules_methods_soft_delete(
    pid: str,
    dataset_id: str,
    table: str,
    module_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Soft delete modules_to_methods links."""
    query = f"""
        UPDATE `{pid}.{dataset_id}.{table}`
        SET status = 'deleted'
        WHERE module_id = @module_id AND user_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_update_modules_methods_links_soft_delete(
    pid: str,
    dataset_id: str,
    table: str,
    module_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Soft delete module-method links (for link_module_methods)."""
    query = f"""
        UPDATE `{pid}.{dataset_id}.{table}`
        SET status = 'deleted'
        WHERE module_id = @module_id AND user_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_rm_link_env_module(
    pid: str,
    dataset_id: str,
    session_id: str,
    env_id: str,
    module_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Soft delete env-to-module link."""
    query = f"""
        UPDATE `{pid}.{dataset_id}.envs_to_modules`
        SET status = 'deleted'
        WHERE session_id = @session_id AND env_id = @env_id AND module_id = @module_id AND user_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", session_id),
            bigquery.ScalarQueryParameter("env_id", "STRING", env_id),
            bigquery.ScalarQueryParameter("module_id", "STRING", module_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_get_env_module_structure_mods(
    pid: str,
    dataset_id: str,
    session_id: str,
    env_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get module_ids for env in session."""
    ds = f"{pid}.{dataset_id}"
    query = f"""
        SELECT module_id FROM `{ds}.envs_to_modules`
        WHERE session_id=@sid AND env_id=@eid AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sid", "STRING", session_id),
            bigquery.ScalarQueryParameter("eid", "STRING", env_id),
        ]
    )
    return query, job_config


def bq_get_env_module_structure_fields(
    pid: str,
    dataset_id: str,
    session_id: str,
    env_id: str,
    module_ids: List[str],
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get fields for modules in env."""
    ds = f"{pid}.{dataset_id}"
    query = f"""
        SELECT module_id, field_id FROM `{ds}.modules_to_fields`
        WHERE session_id=@sid AND env_id=@eid AND module_id IN UNNEST(@mids)
        AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sid", "STRING", session_id),
            bigquery.ScalarQueryParameter("eid", "STRING", env_id),
            bigquery.ArrayQueryParameter("mids", "STRING", module_ids),
        ]
    )
    return query, job_config


def bq_retrieve_logs_env(
    pid: str,
    dataset_id: str,
    env_id: str,
    user_id: str,
    limit: int = 100,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get logs for env."""
    query = f"""
        SELECT timestamp, message
        FROM `{pid}.{dataset_id}.logs`
        WHERE env_id = @env_id AND user_id = @user_id
        ORDER BY timestamp DESC
        LIMIT {limit}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("env_id", "STRING", env_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_get_env_data(
    pid: str,
    dataset_id: str,
    env_id: str,
    limit: int = 50,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get sim_data for env."""
    query = f"""
        SELECT *
        FROM `{pid}.{dataset_id}.sim_data`
        WHERE env_id = @env_id
        ORDER BY created_at DESC
        LIMIT {limit}
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("env_id", "STRING", env_id)]
    )
    return query, job_config


def bq_get_user(
    pid: str,
    dataset_id: str,
    uid: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get user by id."""
    query = f"""
        SELECT * FROM `{pid}.{dataset_id}.users`
        WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
    )
    return query, job_config


def bq_get_payment_record(
    pid: str,
    dataset_id: str,
    uid: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get payment record by user id."""
    query = f"""
        SELECT * FROM `{pid}.{dataset_id}.payment`
        WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
    )
    return query, job_config


def bq_get_standard_stack(
    pid: str,
    dataset_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get user sm_stack_status."""
    query = f"""
        SELECT * from `{pid}.{dataset_id}.users`
        WHERE id = @user_id AND (status != 'deleted' OR status IS NULL)
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
    )
    return query, job_config


def bq_ensure_user_exists(
    pid: str,
    dataset_id: str,
    uid: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Check if user exists."""
    query = f"""
        SELECT id FROM `{pid}.{dataset_id}.users`
        WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
    )
    return query, job_config


def bq_ensure_payment_exists(
    pid: str,
    dataset_id: str,
    uid: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Check if payment record exists."""
    query = f"""
        SELECT id FROM `{pid}.{dataset_id}.payment`
        WHERE id = @uid AND (status != 'deleted' OR status IS NULL)
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
    )
    return query, job_config


def bq_get_session_envs(
    ds_ref: str,
    user_id: str,
    session_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get envs linked to session."""
    query = f"""
        SELECT env_id, user_id, status
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY env_id ORDER BY created_at DESC) as row_num
            FROM `{ds_ref}.session_to_envs`
            WHERE user_id = @user_id AND session_id = @session_id
        )
        WHERE row_num = 1 AND status != 'deleted'
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("session_id", "STRING", str(session_id)),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_get_active_session(
    ds_ref: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get active session for user."""
    query = f"""
        SELECT id FROM `{ds_ref}.sessions`
        WHERE user_id = @user_id AND is_active = TRUE AND status != 'deleted'
        ORDER BY created_at DESC
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
    )
    return query, job_config


def bq_get_full_session_envs(
    pid: str,
    dataset_id: str,
    session_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get env_ids for session (full structure)."""
    query = f"""
        SELECT env_id FROM `{pid}.{dataset_id}.session_to_envs`
        WHERE session_id=@sid AND user_id=@uid AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sid", "STRING", session_id),
            bigquery.ScalarQueryParameter("uid", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_get_full_session_modules(
    pid: str,
    dataset_id: str,
    session_id: str,
    env_ids: List[str],
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get modules for envs in session."""
    query = f"""
        SELECT env_id, module_id FROM `{pid}.{dataset_id}.envs_to_modules`
        WHERE session_id=@sid AND env_id IN UNNEST(@eids) AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sid", "STRING", session_id),
            bigquery.ArrayQueryParameter("eids", "STRING", env_ids),
        ]
    )
    return query, job_config


def bq_get_full_session_fields(
    pid: str,
    dataset_id: str,
    session_id: str,
    env_ids: List[str],
    module_ids: List[str],
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get fields for modules in session."""
    query = f"""
        SELECT env_id, module_id, field_id FROM `{pid}.{dataset_id}.modules_to_fields`
        WHERE session_id=@sid AND env_id IN UNNEST(@eids) AND module_id IN UNNEST(@mids)
        AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("sid", "STRING", session_id),
            bigquery.ArrayQueryParameter("eids", "STRING", env_ids),
            bigquery.ArrayQueryParameter("mids", "STRING", module_ids),
        ]
    )
    return query, job_config


def bq_get_field_interactants(
    pid: str,
    dataset_id: str,
    table: str,
    field_id: str,
    select: str = "*",
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get field interactants."""
    query = f"""
        SELECT {select} FROM `{pid}.{dataset_id}.{table}`
        WHERE field_id = @field_id AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("field_id", "STRING", str(field_id))]
    )
    return query, job_config


def bq_retrieve_session_fields(
    pid: str,
    dataset_id: str,
    table: str,
    module_ids: List[str],
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get distinct field_ids for modules."""
    query = f"""
        SELECT DISTINCT field_id FROM `{pid}.{dataset_id}.{table}`
        WHERE module_id IN UNNEST(@module_ids) AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("module_ids", "STRING", module_ids),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_get_fields_params(
    pid: str,
    dataset_id: str,
    table: str,
    field_id: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get params linked to field."""
    query = f"""
        SELECT * FROM `{pid}.{dataset_id}.{table}`
        WHERE field_id = @field_id AND user_id = @user_id AND (status != 'deleted' OR status IS NULL)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("field_id", "STRING", field_id),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config


def bq_select_table_limit(
    pid: str,
    ds_id: str,
    table_name: str,
    limit: int = 100,
) -> str:
    """SELECT * FROM table LIMIT n."""
    return f"SELECT * FROM `{pid}.{ds_id}.{table_name}` LIMIT {limit}"


def bq_select_distinct_column(
    pid: str,
    ds_id: str,
    table: str,
    column: str,
) -> str:
    """SELECT DISTINCT column FROM table."""
    return f"SELECT DISTINCT {column} FROM `{pid}.{ds_id}.{table}`"


def bq_select_by_id(
    pid: str,
    ds_id: str,
    table_id: str,
    target_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """SELECT * WHERE id = target_id. Use parameterized for safety."""
    query = f"SELECT * FROM `{pid}.{ds_id}.{table_id}` WHERE id = @target_id"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("target_id", "STRING", target_id)]
    )
    return query, job_config


def bq_select_table_all(table_ref: str) -> str:
    """SELECT * FROM table_ref."""
    return f"SELECT * FROM `{table_ref}`"


def bq_extract_schema_sample_value(
    table_ref: str,
    column_name: str,
    limit: int = 1,
) -> str:
    """SELECT one non-null value for column (schema extraction). Validate column_name before use."""
    return f"SELECT `{column_name}` FROM `{table_ref}` WHERE `{column_name}` IS NOT NULL LIMIT {limit}"


def bq_create_or_replace_table(
    pid: str,
    dataset_id: str,
    table_name: str,
    cols_sql: str,
) -> str:
    """CREATE OR REPLACE TABLE with columns."""
    return f"""
        CREATE OR REPLACE TABLE `{pid}.{dataset_id}.{table_name}` (
          {cols_sql}
        )
    """


# ---------- BQ Auth Handler (parameterized) ----------

def bq_auth_check_user_exists(
    pid: str,
    ds_id: str,
    table: str,
    email: str,
    password: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Check if user exists by email and password."""
    query = f"""
        SELECT user_id FROM `{pid}.{ds_id}.{table}`
        WHERE email = @email AND password = @password
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("email", "STRING", email),
            bigquery.ScalarQueryParameter("password", "STRING", password),
        ]
    )
    return query, job_config


def bq_auth_get_user(
    pid: str,
    ds_id: str,
    table: str,
    email: str,
    password: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get user by email and password."""
    query = f"""
        SELECT * FROM `{pid}.{ds_id}.{table}`
        WHERE email = @email AND password = @password
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("email", "STRING", email),
            bigquery.ScalarQueryParameter("password", "STRING", password),
        ]
    )
    return query, job_config


def bq_auth_get_user_from_email(
    pid: str,
    ds_id: str,
    table: str,
    email: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get user by email."""
    query = f"""
        SELECT * FROM `{pid}.{ds_id}.{table}`
        WHERE email = @email
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("email", "STRING", email)]
    )
    return query, job_config


def bq_auth_get_user_from_id(
    pid: str,
    ds_id: str,
    table: str,
    uid: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Get user by user_id."""
    query = f"""
        SELECT * FROM `{pid}.{ds_id}.{table}`
        WHERE user_id = @uid
        LIMIT 1
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("uid", "STRING", uid)]
    )
    return query, job_config


def bq_auth_delete_user(
    pid: str,
    ds_id: str,
    table: str,
    user_id: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """Delete user by user_id."""
    query = f"""
        DELETE FROM `{pid}.{ds_id}.{table}`
        WHERE user_id = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("user_id", "STRING", user_id)]
    )
    return query, job_config


def bq_auth_update_user_field(
    pid: str,
    ds_id: str,
    table: str,
    user_id: str,
    field_name: str,
    new_value: str,
) -> Tuple[str, bigquery.QueryJobConfig]:
    """UPDATE user field. field_name must be validated (whitelist) before use."""
    query = f"""
        UPDATE `{pid}.{ds_id}.{table}`
        SET `{field_name}` = @new_value
        WHERE user_id = @user_id OR email = @user_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("new_value", "STRING", str(new_value)),
            bigquery.ScalarQueryParameter("user_id", "STRING", user_id),
        ]
    )
    return query, job_config

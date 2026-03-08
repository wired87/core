# Prompt: BigQuery Toolbox - Table Filter

**Source**: `qbrain/_bigquery_toolbox/prompts.py` – `get_table_filter_prompt`

User Query: `{user_input}`

Available Tables: `{all_tables}`

Select the tables that are likely to contain information RELEVANT to the user's query.
If the query is generic, select the most important core tables (like 'nodes' or 'edges').

Return a JSON list of strings, e.g. `["table1", "table2"]`.

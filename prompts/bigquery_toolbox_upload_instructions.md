# Prompt: BigQuery Toolbox - Upload Instructions

**Source**: `qbrain/_bigquery_toolbox/prompts.py` – `get_upload_instructions_text`

To add a table or ingest data, please use the **Ingest** command in the CLI.

Example:
```
python cli.py ingest --chunk-size 1000 --use-docai
```

Or ensure your files are in `data_dir` and ask me to "ingest data" if configured.

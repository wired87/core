# Prompt: Session Manager Research Files Column

Within session manager implement a `research_files` column which receives a list[str] urls param.

Use the `self.session_id` and `user_id` params to receive all entries from the upserted list.

Locally merge both lists and overwrite the specific column with the result.

Implement this method within the orchestrator research thread callback method workflow.

# qbrain package contents

Collected from project root. **Excluded:** README.md, _admin (bob_builder, _ray_core), MiracleAI, jax_test, qdash, LICENSE, and general project files (.gitignore, .env, manage.py, Dockerfile, startup.sh, startup.py, r.txt, .git, .idea, .cursor, .venv, node_modules, static_root, media).

## Directories

- `app_handler/`
- `auth/`
- `a_b_c/`
- `chat_manger/`
- `cloud_run/`
- `code_manipulation/`
- `compute_engine/`
- `core/`
- `data/`
- `deploy_training_job/`
- `docs/`
- `done/`
- `embedder/`
- `gem_core/`
- `graph/`
- `grid/`
- `nginx/`
- `qf_utils/`
- `utils/`
- `views/`
- `workflows/`
- `_bigquery_toolbox/`
- `_cloud_run/`
- `_db/`
- `_gmail/`
- `_god/`

## Top-level files

- `create_env.py`
- `get_data.py`
- `main.py`
- `predefined_case.py`
- `relay_station.py`
- `run_daphne.py`
- `type.py`
- `urls.py`
- `visualize.py`
- `vm_init.py`
- `wastlands.py`
- `ws_sim_helper.py`
- `ws_test.py`
- `fix_guard.py`
- `model_output.py`
- `test_filemanager_format.py`
- `ds.md`
- `tasks.md`

## Populating this package

To copy all listed dirs and files into `qbrain/` (so the package is self-contained), run from project root:

```python
from _admin.bob_builder.qbrain_collector import collect_into_qbrain
collect_into_qbrain(copy=True)
```

To move instead of copy (and then update imports to `qbrain.*`):

```python
collect_into_qbrain(copy=False)
```

To only list what would be collected:

```python
from _admin.bob_builder.qbrain_collector import list_collectible
print(list_collectible())
```

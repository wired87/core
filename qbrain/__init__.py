"""
qbrain package: collected application code (core, bm, auth, graph, etc.).

Contains all project dirs and files except:
- README.md, LICENSE (and general project files like .gitignore, .env, manage.py, Dockerfile, startup.*)
- _admin (bob_builder, _ray_core), MiracleAI, jax_test, qdash

To populate this package from project root, run:
    from _admin.bob_builder.qbrain_collector import collect_into_qbrain
    collect_into_qbrain(copy=True)

Or list what would be collected:
    from _admin.bob_builder.qbrain_collector import list_collectible
    print(list_collectible())
"""

__all__ = [
    "app_handler",
    "auth",
    "a_b_c",
    "chat_manger",
    "cloud_run",
    "code_manipulation",
    "compute_engine",
    "core",
    "data",
    "deploy_training_job",
    "docs",
    "done",
    "embedder",
    "gem_core",
    "graph",
    "grid",
    "nginx",
    "qf_utils",
    "utils",
    "views",
    "workflows",
    "_bigquery_toolbox",
    "_cloud_run",
    "_db",
    "_gmail",
    "_god",
    "create_env",
    "get_data",
    "main",
    "model_output",
    "predefined_case",
    "relay_station",
    "run_daphne",
    "type",
    "urls",
    "visualize",
    "vm_init",
    "wastlands",
    "ws_sim_helper",
    "ws_test",
    "fix_guard",
    "test_filemanager_format",
]

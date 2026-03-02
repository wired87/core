Run or include in _admin: start `cd qdash && npm start` → serve the application smoothly with a nice order to push buttons and walk through the entire workflow → collect HTML → save as .mp4 (or .webm) in project root.

Implement local path to OpenAI app creator process: `_admin/app_handler/openai_asdk/config.py` exposes `get_demo_paths()` and `get_demo_video_path()`; `demo_paths.json` (written by `--record-qdash-demo`) is the canonical local path source for the submission workflow.

from __future__ import annotations

import json

import dotenv
dotenv.load_dotenv()


def load_data():
    cfg = open("qbrain/test_out.json", "r").read()
    return json.loads(cfg)




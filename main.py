import json
import os.path
import pprint

import requests

from file.yaml import load_yaml

r"""if __name__ == "__main__":
    input_path = r"C:\Users\wired\OneDrive\Desktop\Projects\bm\gnn\processing\layer\uniprot\cc_pred_ckpt.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
"""

headers = {
    "Content-Type": "application/json"   # <-- MANDATORY
}

if __name__ == "__main__":
    r"""content=load_yaml(r"C:\Users\wired\OneDrive\Desktop\base_dj\betse_app\betse-1.5.0\betse\data\yaml\sim_config.yaml")
    pprint.pp(content)
    r = requests.post("http://127.0.0.1:8000/betse/run/", json={"sim_config_data": content }, headers=headers)
    print(r.json())"""
    print(os.path.relpath("AUTHORS.md", start="betse_app"))
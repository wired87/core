import json

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

import requests

if __name__ == "__main__":
    with open(r"C:\Users\wired\OneDrive\Desktop\base_dj\betse_app\betse-1.5.0\betse\data\yaml\sim_config.yaml", "rb") as f:
        files = {"sim_config_file":  f}
        r = requests.post("https://www.bestbrain.tech/betse/run/", json=files, headers=headers)
        print(r.json())

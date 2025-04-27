import json

import requests


if __name__ == "__main__":
    input_path = r"C:\Users\wired\OneDrive\Desktop\Projects\bm\gnn\processing\layer\uniprot\cc_pred_ckpt.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    with open(input_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)



if __name__ == "__main__":
    """god = CreateGWorld()
    # asyncio.run(god.reinit())
    asyncio.run(god.hello_world())
    world = World(g_utils=god.g_utils)
    world.run_world()"""
    r = requests.post("https://bm2-1004568990634.asia-east1.run.app/sp/mvp", data={"hi": "betse"})
    print(r.json())
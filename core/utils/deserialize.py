import json


def deserialize(world_cfg):
    try:
        if isinstance(world_cfg, str):
            print("deserialize", world_cfg)
            world_cfg = json.loads(world_cfg)
    except Exception as e:
        print(f"Err deserialize: {e}")
    return world_cfg
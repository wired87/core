
import logging
import uuid
import time
from core.handler_utils import flatten_payload
from core.env_manager.env_lib import EnvManager, handle_set_env, handle_get_envs_user, handle_del_env

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_env_manager():
    print("Testing EnvManager...")
    
    # 1. Initialize
    try:
        manager = EnvManager()
        print("EnvManager initialized.")
    except Exception as e:
        print(f"Failed to init EnvManager: {e}")
        return

    # Generate dummy user and env
    user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    env_id = f"test_env_{uuid.uuid4().hex[:8]}"
    
    env_data = {
        "id": env_id,
        "sim_time": 100,
        "cluster_dim": 2,
        "dims": 3
    }

    # 2. Test SET
    print(f"\nTesting SET_ENV for user {user_id}...")
    auth = {"user_id": user_id}
    data = {"env": env_data}
    
    payload = {"auth": auth, "data": data}
    resp = handle_set_env(**flatten_payload(payload))
    print(f"Set Response: {resp}")
    
    # Verify response contains the env
    envs = resp.get("data", {}).get("envs", [])
    found = any(e["id"] == env_id for e in envs)
    if found:
        print("SUCCESS: Env found in response after set.")
    else:
        print("FAILURE: Env NOT found in response after set.")

    # 3. Test GET
    print(f"\nTesting GET_ENV for user {user_id}...")
    resp_get = handle_get_envs_user(**flatten_payload({"auth": auth}))
    envs_get = resp_get.get("data", {}).get("envs", [])
    found_get = any(e["id"] == env_id for e in envs_get)
    if found_get:
        print("SUCCESS: Env found in GET response.")
    else:
        print("FAILURE: Env NOT found in GET response.")

    # 4. Test DELETE
    print(f"\nTesting DEL_ENV for env {env_id}...")
    auth_del = {"user_id": user_id, "env_id": env_id}
    resp_del = handle_del_env(**flatten_payload({"auth": auth_del}))
    print(f"Del Response: {resp_del}")
    
    envs_del = resp_del.get("data", {}).get("envs", [])
    found_del = any(e["id"] == env_id for e in envs_del)
    if not found_del:
        print("SUCCESS: Env gone after delete.")
    else:
        print("FAILURE: Env STILL PRESENT after delete.")

if __name__ == "__main__":
    test_env_manager()

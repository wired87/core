import os
import sys
from qbrain.qf_utils.all_subs import ALL_SUBS
import dotenv

dotenv.load_dotenv()

MAX_GRID_SIZE = 3**3
SCHEMA_GRID = [
    (i, i, i)
    for i in range(MAX_GRID_SIZE)
]

ARSENAL_PATH=r"C:\Users\bestb\PycharmProjects\BestBrain\core\arsenal" if os.name == "nt" else r"core/arsenal"

## GCP
GCP_ID = os.environ.get("GCP_PROJECT_ID")
FBDB_INSTANCE = os.environ.get("FIREBASE_RTDB")
GEM_API_KEY=os.environ.get("GEMINI_API_KEY")
LOGGING_DIR = os.environ.get("LOGGING_DIR")


BASE_MODULES={}
def get_runnables():
    try:
        import jax.numpy as jnp
        import jax
        return {
            "jnp": jnp,
            "jit": jax.jit,
            "jax": jax,
            "vmap": jax.vmap,
        }
    except ImportError:
        import numpy as np
        from unittest.mock import MagicMock
        mock_jax = MagicMock()
        mock_jax.numpy = np
        sys.modules["jax"] = mock_jax
        sys.modules["jax.numpy"] = np
        return {
            "jnp": np,
            "jit": lambda x, *args, **kwargs: x,
            "jax": mock_jax,
            "vmap": lambda x, *args, **kwargs: x,
        }


RUNNABLE_MODULES=get_runnables()

def get_demo_env():
    return {'cluster_dim': [12, 12, 12],
     'cpu': 6,
     'cpu_limit': 6,
     'device': 'cpu',
     'env': [
         {'name': 'SESSION_ID',
          'value': 'env_rajtigesomnlhfyqzbvx_atqbmkasfekqjfgkzrdr'},
         {'name': 'DOMAIN', 'value': 'bestbrain.tech'},
         {'name': 'GCP_ID', 'value': 'aixr-401704'},
         {'name': 'DATASET_ID', 'value': 'QCOMPS'},
         {'name': 'SIM_LEN_S', 'value': '300'},
         {'name': 'LOGGING_DIR', 'value': 'tmp/ray'},
         {'name': 'ENV_ID',
          'value': 'env_rajtigesomnlhfyqzbvx_atqbmkasfekqjfgkzrdr'},
         {'name': 'USER_ID', 'value': 'rajtigesomnlhfyqzbvx'},
         {'name': 'FIREBASE_RTDB',
          'value': 'https://bestbrain-39ce7-default-rtdb.firebaseio.com/'},
         {'name': 'FB_DB_ROOT',
          'value': 'users/rajtigesomnlhfyqzbvx/env/env_rajtigesomnlhfyqzbvx_atqbmkasfekqjfgkzrdr'},
         {'name': 'DELETE_POD_ENDPOINT', 'value': 'gke/delete-pod/'},
         {'name': 'GKE_SIM_CLUSTER_NAME', 'value': 'sims'}
     ],
     'gpus': 0,
     'mem': '12Gi',
     'mem_limit': '12Gi',
     'sim_time': 300
}
USER_ID = os.environ.get("USER_ID")
ENV_ID = os.environ.get("ENV_ID")
# sg
SG_IID = os.environ.get("ENV_ID")
SG_DBID = os.environ.get("SG_DB_ID")
GNAME = os.environ.get("ENV_ID")
DB_NAME=USER_ID
TABLE_NAME=ENV_ID

# ENV VARS
DOMAIN = os.environ.get("DOMAIN")
MODULE_PATH=os.environ.get("MODULE_PATH", r"sm")
SESSION_ID = os.environ.get("SESSION_ID")


# VARS
TESTING = True
FB_DB_ROOT = f"users/{USER_ID}/env/{ENV_ID}"
DEMO_ENV=get_demo_env()
HEAD_SERVER_NAME = "HEAD"
ENDPOINTS = [*ALL_SUBS, "EDGES"]
USED_ENDPOINTS = []
LOCAL_DATASTORE = False

JAX_DEVICE="cpu" if TESTING is True else "gpu"

GPU=os.getenv("GPU")

# SET ENVS
os.environ["APP_ROOT"] = os.path.dirname(os.path.abspath(sys.argv[0]))

NUM_GPU_TOTAL = 0 if TESTING is True else 1
NUM_GPU_NODE = 0 if TESTING is True else .33

NUM_CPU_TOTAL = 4

###########################
###### CLASSES

# LOGIC (related) VARS
if os.name == "nt":
    trusted = ["*"]
else:
    trusted=[
        f"{DOMAIN}.com", f"*.{DOMAIN}.com", "localhost", "127.0.0.1"]

ALL_DB_WORKERS=[
    "FBRTDB",
    "SPANNER_WORKER",
    "BQ_WORKER",
]


SIMULATE_ON_QC=os.getenv("SIMULATE_ON_QC")
if str(SIMULATE_ON_QC) == "0":
    SIMULATE_ON_QC = True
else:
    SIMULATE_ON_QC = False


GLOBAC_STORE = {
    key: None
    for key in [
        "UTILS_WORKER",
        "DB_WORKER",
        "HEAD",
        "GLOB_LOGGER",
        "GLOB_STATE_HANDLER",
        "BQ_WORKER",
        "SPANNER_WORKER",
        "WEB_DATA_PROVIDER",
    ]
}

def extend_globs(key, value):
    GLOBAC_STORE[key] = value
    print(f"EXTEND GLOB STORE WITH {key}={value}")

def get_endpoint():
    for endp in ENDPOINTS:
        if endp not in USED_ENDPOINTS:
            USED_ENDPOINTS.append(endp)
            return endp





SHIFT_DIRS = [
     [
        [ 1, 0, 0], [0,  1, 0], [0, 0,  1],
        [ 1, 1, 0], [1, 0,  1], [0, 1,  1],
        [ 1,-1, 0], [1, 0, -1], [0, 1, -1],
        [ 1, 1, 1], [1, 1,-1], [1,-1, 1], [1,-1,-1],
     ],
     [
        [-1, 0, 0], [0, -1, 0], [0, 0, -1],
        [-1,-1, 0], [-1, 0,-1], [0,-1,-1],
        [-1, 1, 0], [-1, 0, 1], [0,-1, 1],
        [-1,-1,-1], [-1,-1, 1], [-1, 1,-1], [-1, 1, 1],
    ]
]



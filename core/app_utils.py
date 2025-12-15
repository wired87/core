import os
import sys

import jax
from jax import jit, vmap
import jax.numpy as jnp

from fastapi import FastAPI

from _god.env_node_creator import EnvNodeCreator
from qf_utils.all_subs import ALL_SUBS

import dotenv
import cfg

dotenv.load_dotenv()

RELAY_PKG_CONFIG = cfg.RELAY_CONFIG

# todo global nxGs and classes
APP = FastAPI()

ARSENAL_PATH=r"C:\Users\bestb\Desktop\qfs\arsenal" if os.name == "nt" else r"/arsenal"

## GCP
GCP_ID = os.environ.get("GCP_ID")
FBDB_INSTANCE = os.environ.get("FIREBASE_RTDB")
GEM_API_KEY=os.environ.get("GEMINI_API_KEY")
LOGGING_DIR = os.environ.get("LOGGING_DIR")
RUNNABLE_MODULES={
    "jnp": jnp,
    "jit": jit,
    "jax": jax,
    "vmap": vmap,
}

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
     'sim_time_s': 300
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
MODULE_PATH=os.environ.get("MODULE_PATH", r"/sm")
SESSION_ID = os.environ.get("SESSION_ID")



RANGE = os.environ.get("RANGE")
RANGE = range(
    int(RANGE.split("-")[0]),
    int(RANGE.split("-")[1])
)

EXEC_SCPE={
    "jit": jit
}

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


SHIFT_DIRS={
    "x": (1, 0, 0),
    "y": (0, 1, 0),
    "z": (0, 0, 1),
    "xy_pp": (1, 1, 0),
    "xy_pm": (1, -1, 0),
    "xz_pp": (1, 0, 1),
    "xz_pm": (1, 0, -1),
    "yz_pp": (0, 1, 1),
    "yz_pm": (0, 1, -1),
    "xyz_ppp": (1, 1, 1),
    "xyz_ppm": (1, 1, -1),
    "xyz_pmp": (1, -1, 1),
    "xyz_pmm": (1, -1, -1)
}


ENVC=EnvNodeCreator(env_id=ENV_ID, world_cfg=None).create()
DIM=3
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









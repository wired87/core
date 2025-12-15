import numpy as np
import ray
import os

from _ray_core.base.admin_base import RayAdminBase
from _ray_core.globacs.glob_creator import GlobsMaster

from app_utils import TESTING, USER_ID, ENV_ID, DEMO_ENV
from _ray_core.globacs.state_handler.main import StateHandler
from fb_core.real_time_database import FBRTDBMgr
from project_creators import start_relay, create_db_swat

from server import Server

OS_NAME = os.name
STATE_HANDLER = StateHandler()
RELAY = None

def fetch_world_content():
    if TESTING is False:
        world_cfg_mgr = FBRTDBMgr()
        world_cfg = world_cfg_mgr.get_data(
            path=f"users/{USER_ID}/env/{ENV_ID}/cfg/world/",
            child=True,
        )

        if world_cfg is None:
            print("no wcfg found. exit.")
            return
        world_cfg = world_cfg["world"]

    else:
        print("RETRUN DEMO ENV:", DEMO_ENV)
        world_cfg = DEMO_ENV

    # MANIPULATE WORLD CFG
    if TESTING is True:
        world_cfg["cluster_dim"] = 2
        world_cfg["particle"] = "electron"
        world_cfg["phase"] = 3
        world_cfg["sim_time_s"] = 1
        world_cfg["energy"] = 20

    # calc 3D amount nodes
    world_cfg["amount_nodes"] = np.prod([world_cfg["cluster_dim"] for _ in range(3)])
    return world_cfg


def create_globs_worker(world_cfg):
    # Create Lexi -> load arsenal
    utils = GlobsMaster.options(
        lifetime="detached",
        name="GLOB_MASTER"
    ).remote(
        world_cfg=world_cfg,
    )
    ray.get(utils.create.remote())



def main():
    world_cfg = fetch_world_content()

    if world_cfg:
        create_globs_worker(world_cfg)

        create_db_swat(world_cfg)

        start_relay(
            world_cfg,
        )

        print("Project initialized")

        if TESTING is True:
            server = Server()
            server.run()

if __name__ == "__main__":
    main()

import ray
resources=[
    "FBRTDB",
    "BQ_WORKER",
]



def start_relay(world_cfg):
    from relay import Relay
    print("START RELAY")
    name = "RELAY"
    ref = Relay.options(
        name=name,
        lifetime="detached",
    ).remote(
        world_cfg=world_cfg,
    )

    ray.get_actor(name="UTILS_WORKER").set_node.remote(
        dict(
            id=name,
            ref=ref._ray_actor_id.binary().hex(),
            type="ACTOR"
        )
    )

    ref.prepare.remote()
    print("RELAY STARTED")



def create_db_swat(world_cfg):
    from a_b_c import CloudMaster
    ref = CloudMaster.options(
        lifetime="detached",
        name="CLOUD_MASTER"
    ).remote(
        world_cfg=world_cfg,
        resources=resources,
    )
    ray.get(ref.create_actors.remote())




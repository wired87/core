from ray.actor import ActorHandle
from ray.serve.handle import DeploymentHandle

# DBWorker Get set on node lvl
HOST_TYPE = {
    "head": DeploymentHandle or None,
    "qfn": ActorHandle or DeploymentHandle or None,
    "db_worker": ActorHandle or None,
    "field_worker": ActorHandle or None
}

WS_INBOUND = {
    "action": str,
    "key": str,
}


WS_OUTBOUND = {
    "type": str,
    "key": str,
    "data": dict,
}

LISTENER_PAYLOAD= {
    "type":str,
    "path": str,
    "data": dict
}
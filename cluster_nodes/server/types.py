from typing import Dict, Union

from ray.actor import ActorHandle
from ray.serve.handle import DeploymentHandle

HOST_TYPE = Dict[str, Union[str, DeploymentHandle, ActorHandle]]
#{id, handle)
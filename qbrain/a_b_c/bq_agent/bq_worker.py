import ray

from qbrain.a_b_c.bq_agent._bq_core.loader.aloader import ABQHandler
from qbrain.a_b_c.bq_agent._bq_core.manager_wrapper import BQController

from qbrain.core.app_utils import DB_NAME
from _admin._ray_core.base.base import BaseActor


@ray.remote(num_cpus=0.2)
class BQService(
    BaseActor,
    BQController
):
    def __init__(self):
        BaseActor.__init__(self)
        BQController.__init__(self)



if __name__ == "__main__":
    abq = ABQHandler(dataset=DB_NAME)
    abq.bq_insert(
        table_id="TEST_TABLE2011",
        rows=[{
            "id": "test_nid",
            "key": [12454]
        }]
    )


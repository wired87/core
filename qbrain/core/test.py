from qbrain._god.create_world import God
from qbrain.core.app_utils import USER_ID, ENV_ID
from main import fetch_world_content
from qbrain.qf_utils.qf_utils import QFUtils
from qbrain.qf_utils.runtime_utils_creator import RuntimeUtilsCreator
from qbrain.graph.local_graph_utils import GUtils

g=GUtils()
qfu=QFUtils(g)

world_cfg = fetch_world_content()

god = God(
    G=g.G,
    qfu=qfu,
    env_id=ENV_ID,
    user_id=USER_ID,
    world_cfg=world_cfg
)

god.create_world()

ruc = RuntimeUtilsCreator(
    g=g,
)

# admin_data processor -> get single file
for nid, attrs in g.G.nodes(data=True):
    pass




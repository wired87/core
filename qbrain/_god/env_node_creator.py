from qbrain.qf_utils.field_utils import FieldUtils


class EnvNodeCreator:

    def __init__(
        self,
        world_cfg=None,
    ):
        self.world_type = "bare"
        self.world_cfg = world_cfg
        self.content_base_path = r"C:\Users\wired\OneDrive\Desktop\Projects\Brainmaster\utils\simulator\world\env"
        self.fu = FieldUtils()
        self.layer = "ENV"
        self.ion_count = 0
        print("ENVCCreator initialized")






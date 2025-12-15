from data import ENV
from qf_utils.field_utils import FieldUtils


class EnvNodeCreator(FieldUtils):

    def __init__(
            self,
            env_id,
            world_cfg=None,
    ):
        FieldUtils.__init__(self)
        self.world_type = "bare"
        self.world_cfg = world_cfg
        self.content_base_path = r"C:\Users\wired\OneDrive\Desktop\Projects\Brainmaster\utils\simulator\world\env"

        self.env_id = env_id
        self.layer = "ENV"
        self.ion_count = 0
        print("ENVCCreator initialized")

    def validate_env_type(self):
        env_content = None

        if self.world_type == "bare":
            # bare physics simulation
            env_content = ENV.copy()
        elif self.world_type == "cellular":
            # biophysic sim
            pass
        return env_content


    def create(self):
        """
        Create an ENVC layer (Environment Control Layer)
        which manages global fields/states like temperature, etc.
        """
        # todo customize settings -> read from received yaml -> incl tiemsteps
        env = dict(
                nid=self.env_id,
                world_type=self.world_type,
                type=self.layer,
                pm_axes=self.get_dirs(),
                parent=["USERS"],
                i=1j,
                **ENV,
            )
        return env

    def get_dirs(self):
        plus = list(self.direction_definitions.values())
        minus = []
        for dir_str, struct in self.direction_definitions.items():
            n_struct = []
            for dir in struct:
                is_negative = dir < 0
                if is_negative:
                    n_struct.append(abs(dir))
                else:
                    n_struct.append(int(dir * -1))
            minus.append(tuple(n_struct))
        axes = [plus, minus]
        return axes

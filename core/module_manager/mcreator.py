import os
from ray import get_actor
from core._ray_core.base.base import BaseActor
from core._ray_core.globacs.state_handler.main import StateHandler
from core.app_utils import ARSENAL_PATH
from core.module_manager.modulator import Modulator
from utils.graph.local_graph_utils import GUtils

import dotenv
dotenv.load_dotenv()

class ModuleCreator(
    StateHandler,
    BaseActor,
):

    """
    Worker for loading, processing and building of single Module
    """

    def __init__(
            self,
            G,
            qfu,
    ):
        """
        attrs: extracted module content in frmat:
        nid=module_name,
        type="MODULE",
        parent=["FILE"],
        content=mcontent # code str
        """
        print("======== INIT MODULE CREATOR ========")
        super().__init__()
        self.g = GUtils(
            G=G
        )
        self.mmap = []
        self.qfu=qfu
        self.arsenal_struct: list[dict] = None
        self.sm=None

    def load_sm(self):
        new_modules = []
        sm = True
        for module_file in os.listdir(ARSENAL_PATH):
            if not self.g.G.has_node(module_file):
                new_modules.append(module_file)
                self.create_modulator(
                    module_file,
                    from_code_file=True,
                    sm=sm
                )
        return new_modules


    def main(self, temp_path):
        print("=========== MODULATOR CREATOR ===========")
        """
        LOOP (TMP) DIR -> CREATE MODULES FORM MEDIA
        """
        if self.sm is None:
            self.load_sm()
            self.sm=True
        # todo load modules form files
        for root, dirs, files in os.walk(temp_path):
            for module in dirs:
                if not self.g.G.has_node(module):
                    self.create_modulator(
                        module,
                        from_code_file=False,
                    )

            for f in files:
                if not self.g.G.has_node(f):
                    self.create_modulator(
                        f,
                        from_code_file=True,
                    )
        print("modules updated")


    def trigger_buildup_all_modules(self, module_names):
        for module in module_names:
            ref = get_actor(module)
            ref.module_build_process.remote()


    def create_modulator(
            self,
            file,
            from_code_file:bool=False,
            sm=False
    ):
        try:
            mid = file.split(".")[0]
            print("CREATE M", mid)

            module_index = len(
                [
                    nid
                    for nid, _ in self.g.get_nodes("type", "MODULE")
                ]
            )

            mref = Modulator(
                G=self.g.G,
                mid=mid,
                qfu=self.qfu,
                module_index=module_index,
            )

            # save ref
            self.g.add_node(
                dict(
                    nid=mid,
                    ref=mref,
                    type="MODULE",
                    path=file,
                    module_index=module_index,
                    file=from_code_file,
                    callable=None,
                    sm=sm,
                    ready=False,
                )
            )

            print("MODULATORS CREATED")
            mref.module_conversion_process(module_index)

            self.g.update_node(
                attrs=dict(
                    nid=mid,
                    ready=True
                )
            )
        except Exception as e:
            print(f"Err create_modulator: {e}")
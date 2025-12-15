import os

import ray
from ray import get_actor

from _ray_core.base.base import BaseActor
from _ray_core.globacs.state_handler.main import StateHandler
from app_utils import USER_ID, ARSENAL_PATH
from module_manager.modulator import Modulator
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

    def __init__(self):
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
            user_id=USER_ID,
            G=self.get_G()
        )

        self.mmap=[]

        self.arsenal_struct: list[dict] = None

        self.relay = get_actor("RELAY")
        self.fmdir = os.getenv('OUTPUTS')

    def load_sm(self):
        new_modules = []

        for module_file in os.listdir(ARSENAL_PATH):
            if not self.g.G.has_node(module_file):
                new_modules.append(module_file)
                self.create_modulator(
                    module_file,
                    from_code_file=True,
                    module_index=new_modules.index(module_file),
                )
        return new_modules

    def main(self, temp_path):
        print("=========== MODULATOR CREATOR ===========")
        """
        LOOP (TMP) DIR -> CREATE MODULES FORM MEDIA
        """
        # from codebase
        new_modules = []

        # todo load modules form files
        for root, dirs, files in os.walk(temp_path):
            for module in dirs:
                if not self.g.G.has_node(module):
                    new_modules.append(module)
                    self.create_modulator(
                        module,
                        from_code_file=False,
                        module_index=new_modules.index(module)
                    )
            for f in files:
                if not self.g.G.has_node(f):
                    new_modules.append(f)
                    self.create_modulator(
                        f,
                        from_code_file=True,
                        module_index=new_modules.index(f),
                    )
        print("modules extracted", new_modules)
        return new_modules


    def trigger_buildup_all_modules(self, module_names):
        for module in module_names:
            ref = get_actor(module)
            ref.module_build_process.remote()


    def create_modulator(self, file, from_code_file:bool=False,module_index=0):
        mid = file.split(".")[0]
        print("CREATE M", mid)
        mref = Modulator.options(
            name=mid,
            lifetime="detached",
        ).remote(
            g=self.g,
            mid=mid,
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
            )
        )
        mref.module_build_process.remote()
        #todo?
        #get_actor("GUARD").
        print("MODULATORS CREATED")
import ast
import inspect
import os
import pprint

from core.app_utils import TESTING, ARSENAL_PATH

from code_manipulation.graph_creator import StructInspector
from qf_utils.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils
from utils.graph.visual import create_g_visual

class ModuleLoader(
    StructInspector
):

    """
    Build single module from path of code

    ust receive mcfg file
    """

    def __init__(
            self,
            G,
            nid,
    ):
        self.id=nid

        self.g = GUtils(G=G)
        self.qfu=QFUtils(g=self.g)

        StructInspector.__init__(self, self.g.G)

        self.modules = {}
        self.device = "gpu" if TESTING is False else "cpu"
        self.module_g_save_path = rf"C:\Users\bestb\PycharmProjects\BestBrain\outputs\module_{self.id}_G.html" if os.name == "nt" else f"outputs/module_{self.id}_G.html"
        self.finished=False


    def finished(self):
        return self.finished


    def load_local_module_codebase(self, code_base):
        """
        LOAD CONVERTED CODE FOR MODULE
        """
        print("====== load_local_module_codebase ======")
        # tdo use Converter to convert files
        try:
            if code_base is None:
                print(f"load codebase for MODULE {self.id}")
                attrs = self.g.G.nodes[self.id]
                if attrs.get("content", None):
                    print(f"code for already saved in G -> continue")
                    return

                print(f"start code xtraction for {self.id}")
                file_name = os.path.join(ARSENAL_PATH, f"{self.id}.py")

                with open(file_name, "r", encoding="latin-1") as file_handle: #latin-1
                    # Use the file_name as the key
                    print("create module name")
                    module_name = self.id.split(".")[0]
                    print("CREATE MODULE:", module_name)
                code_base = file_handle.read()
            else:
                module_name = self.id.split(".")[0]

            self.g.update_node(
                attrs=dict(
                    nid=module_name,
                    type="MODULE",
                    parent=["FILE"],
                    code=code_base # code str
                )
            )
        except Exception as e:
            print("Err load_local_module_codebase:", e)
            print(f"  Module ID: {self.id}")

        print("finished load_local_module_codebase")
        #self.g.print_status_G()


    def create_code_G(self, mid, code=None):
        print("====== create_code_G ======")
        #print(f"DEBUG: self.g is {self.g}")
        try:
            #GET MDULE ATTRS
            mattrs = self.g.G.nodes[mid]
            
            # Check if content exists
            if "code" not in mattrs:
                raise KeyError(f"Module '{mid}' does not have 'content' attribute. Available attributes: {list(mattrs.keys())}")
            
            # CREATE G FROM IT
            self.convert_module_to_graph(
                code_content=mattrs["code"],
                module_name=mid
            )

            #self.g.print_status_G()

            # todo dest_path?=None -> return html -> upload firebase -> fetch and visualizelive frontend
            create_g_visual(
                self.g.G,
                dest_path=self.module_g_save_path
            )
            print("create_code_G finished")
        except Exception as e:
            print(f"Error in create_code_G for module '{mid}':", e)


    def extract_module_classes(self):
        """
        Executes code from self.modules, extracts custom classes using AST inspection,
        and instantiates the first found class for each module.

        Returns:
        The method updates self.modules in place with the class instances.
        """
        print("====== extract_module_classes ======")
        execution_namespace = {}

        for module_name, module_code in self.modules.items():

            # 1. Use AST to find the name of the custom class defined in the code string
            custom_class_names = []
            try:
                tree = ast.parse(module_code)
                # Iterate through the nodes to find ClassDef nodes
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef):
                        custom_class_names.append(node.name)
            except Exception as e:
                # Handle cases where the code string is not valid Python
                print(f"Error parsing AST for {module_name}: {e}")
                continue

            if not custom_class_names:
                print(f"No custom class found in {module_name}. Skipping execution.")
                continue

            # 2. Execute the code to populate the namespace
            # We only execute if we know there is a class to instantiate
            exec(module_code, execution_namespace)

            # 3. Instantiate the first found class
            # We rely on the AST name being correct
            target_class_name = custom_class_names[0]

            # Check if the class object is actually in the namespace after execution
            if target_class_name in execution_namespace:
                obj = execution_namespace[target_class_name]

                # Final check to ensure it's a class
                if inspect.isclass(obj):
                    try:
                        # Create an instance of the class
                        instance = obj()

                        # Store the instance in self.modules
                        self.modules[module_name] = instance

                    except TypeError as e:
                        # Handle cases where the constructor requires arguments
                        print(
                            f"Cannot instantiate class {target_class_name} from {module_name}: {e}. Constructor requires arguments.")
                else:
                    # This should not happen if the AST parse was correct
                    print(f"Object {target_class_name} in {module_name} is not a class after execution.")
            # Clear the namespace for the next iteration to prevent cross-module interference
            execution_namespace.clear()
        print("finished extract_module_classes")





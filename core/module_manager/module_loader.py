import ast
import inspect
import os

from ray import get_actor

from app_utils import TESTING, USER_ID
from fb_core.real_time_database import FBRTDBMgr
from graph_visualizer.pyvis_visual import create_g_visual
from code_manipulation.graph_creator import StructInspector
from qf_utils.qf_utils import QFUtils
from utils.graph.local_graph_utils import GUtils

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
            fields:list[str],
    ):
        self.id=nid
        self.fields=fields

        self.g = GUtils(USER_ID, G=G)
        self.qfu=QFUtils(g=self.g)

        StructInspector.__init__(self, self.g.G)

        self.module_file_path = self.g.G.nodes[self.id]["path"]

        self.modules = {}
        self.device = "gpu" if TESTING is False else "cpu"
        self.module_g_save_path = rf"C:\Users\bestb\Desktop\qfs\outputs\module_{self.id}_G.html" if os.name == "nt" else "outputs/module_{self.id}_G.html"
        self.firebase = FBRTDBMgr()
        self.finished=False
        self.relay = get_actor("RELAY")


    def finished(self):
        return self.finished


    def load_local_module_codebase(self):
        """
        RECOGNIZE ALL CREATED MODULES
        """
        print("====== load_local_module_codebase ======")
        # tdo use Converter to convert files
        try:

            with open(self.module_file_path, "r", encoding="latin-1") as file_handle: #latin-1
                # Use the file_name as the key
                print("create module name")
                module_name = self.id.split(".")[0]

                print("load modules with module:", module_name)
                mcontent = file_handle.read() # base64.b64encode().decode('utf-8')
                self.g.update_node(
                    dict(
                        nid=module_name,
                        type="MODULE",
                        parent=["FILE"],
                        content=mcontent # code str
                    )
                )

        except Exception as e:
            print("Err load_local_module_codebase:", e)
        print("finished load_local_module_codebase")



    def create_code_G(self, mid):
        print("====== create_code_G ======")
        #GET MDULE ATTRS
        mattrs = self.g.G.nodes[mid]

        # CREATE G FROM IT
        self.convert_module_to_graph(
            code_content=mattrs["content"],
            module_name=mid
        )

        self.g.print_status_G(nid="CODE G")

        # todo dest_path?=None -> return html -> upload firebase -> fetch and visualizelive frontend
        create_g_visual(
            self.g.G,
            dest_path=self.module_g_save_path
        )
        print("create_code_G finished")


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



    def set_arsenal_struct(self) -> list[dict]:
        """
        Gather parent MODULE (e.g. fermion)
        -> get child methods
        -> sort from
        """
        try:
            # get all methods for the desired task
            print("start retrieve_arsenal_struct")
            arsenal_struct: list[dict] = []

            methods: dict[str, dict] = self.g.get_neighbor_list(
                node=self.id,
                target_type="PARAM",
            )

            print("defs found:", methods.keys())

            if methods:
                for i, (mid, attrs) in enumerate(methods.items()):
                    if "__init__" in mid or "main" in mid or mid.split(".")[-1].startswith("_"):
                        continue

                    params = self.g.get_neighbor_list_rel(
                        node=mid,
                        trgt_rel="requires_param",
                        as_dict=True,
                    )

                    print("params set")
                    pkeys = list(params.keys())
                    attrs["params"] = pkeys
                    arsenal_struct.append(attrs)

            # sort
            self.arsenal_struct: list[dict] = self.get_execution_order(
                method_definitions=arsenal_struct
            )

            self.set_method_exec_index()

        except Exception as e:
            print(f"Error RELAY.retrieve_arsenal_struct: {e}")
            return []


    def set_method_exec_index(self):
        # add method index to node to ensure persistency in equaton handling
        for i, item in enumerate(self.arsenal_struct):
            self.g.update_node(
                dict(
                    nid=item["nid"],
                    method_index=i,
                )
            )
        print("finished retrieve_arsenal_struct")






    def get_execution_order(
            self,
            method_definitions: list[dict]
    ) -> list[dict]:
        """
        Determines the correct execution order of methods based on data dependencies.

        Args:
            method_definitions: List of dictionaries, each describing a method:
            {'method_name': str, 'return_key': str, 'parameters': List[str]}

        Returns:
            List[str]: The method names in the required dependency order.
        """

        # Identify all keys that are returned/produced by *any* method in the list.
        internal_returns = {m['return_key'] for m in method_definitions if m.get('return_key')}

        scheduled_order = []
        # Tracks keys that have been produced by methods already scheduled.
        produced_keys = set()

        # Use a mutable copy of the input list for processing.
        remaining_methods = method_definitions

        # Loop until all methods are scheduled or a dependency cycle is found.
        while remaining_methods:
            ready_to_run = []

            # 1. Identify all methods ready in this iteration
            for method in remaining_methods:
                required_params = set(method.get('parameters', []))

                # Dependencies are the internal keys required by the method.
                # External initial inputs (like 'mass', 'vev') are ignored here,
                # as they are assumed to be always available from the start.
                internal_dependencies = required_params.intersection(internal_returns)

                # Check if all required internal dependencies are met by the produced keys.
                if internal_dependencies.issubset(produced_keys):
                    ready_to_run.append(method)

            # Break if no method can run (indicates a cycle or incomplete definition)
            if not ready_to_run:
                break

            # 2. Schedule the ready methods and update the state
            for method in ready_to_run:
                scheduled_order.append(
                    method
                )

                # Update the set of produced keys
                if method.get('return_key'):
                    produced_keys.add(method['return_key'])

                # Remove the method from the remaining list
                remaining_methods.remove(method)

        return scheduled_order

if __name__ == "__main__":
    mm = ModuleLoader(GUtils(USER_ID))
    mm.module_workflow()


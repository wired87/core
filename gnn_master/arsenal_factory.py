import ast
from typing import Dict, Any, Union, Set, Optional

from utils.graph.local_graph_utils import GUtils


def _get_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str:
    """Extracts docstring from function or class node."""
    return ast.get_docstring(node) or ""


def _get_type_name(node: Optional[ast.expr]) -> str:
    """Extracts type name from annotation or defaults to 'Any'."""
    return ast.unparse(node) if node else 'Any'


# --- Step 1: Data Type Definition ---

# Common built-in types and array-like dimension placeholders
BUILTIN_TYPES: Set[str] = {
    'str', 'int', 'float', 'bool', 'list','tuple', 'Any',
    'List', 'Tuple', 'array',  # Common array/dimension concepts
}

class StructInspector(ast.NodeVisitor):

    """
    Traverses AST to populate a CodeGraph with classes, methods, and variables.
    The graph stores the entire structure; no redundant internal dicts are kept.
    """

    def __init__(self, g_utils_instance: Any):
        self.current_class: Optional[str] = None
        self.g = g_utils_instance  # GUtils instance is passed
        self.init_data_type_nodes()

    def init_data_type_nodes(self):
        """Creates nodes for all predefined built-in and array types."""
        for data_type in BUILTIN_TYPES:
            # Node ID is the type name itself
            self.g.add_node(
                node_id=data_type,
                node_type='DATATYPE',
                data={'name': data_type}
            )

    # A. Visit Class
    def visit_ClassDef(self, node: ast.ClassDef):
        """Adds a CLASS node and sets the context for nested methods/variables."""
        class_name = node.name
        self.current_class = class_name

        # 1. Add Class Node
        docstring = _get_docstring(node)
        self.g.add_node(class_name, 'CLASS', {'name': class_name, 'docstring': docstring})

        self.generic_visit(node)
        self.current_class = None

    # B. Visit Methods (Sync/Async)
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """Processes methods (or standalone functions, if not in class)."""
        is_method = bool(self.current_class)
        if not is_method:
            return  # Only process methods for class structure graph

        method_name = node.name
        method_id = f"{self.current_class}.{method_name}"

        # Get Method Data
        return_type = _get_type_name(node.returns)
        docstring = _get_docstring(node)

        # 1. Add METHOD Node
        self.g.add_node(method_id, 'METHOD', {
            'name': method_name,
            'class': self.current_class,
            'returns': return_type,
            'docstring': docstring,
            'is_async': isinstance(node, ast.AsyncFunctionDef)
        })

        # 2. Edge: Class -> Method
        self.g.add_edge(
            self.current_class,
            method_id,
            'has_method'
        )

        # 3. Process Parameters
        for arg in node.args.args:
            if arg.arg == 'self': continue

            param_name = arg.arg
            param_type = _get_type_name(arg.annotation)
            param_id = f"{method_id}.{param_name}"

            # Add PARAM Node
            self.g.add_node(param_id, 'PARAM', {'name': param_name, 'type': param_type})

            # Edge: Method -> Parameter (Needs Input)
            self.g.add_edge(method_id, param_id, 'needs_param', {'type': param_type})

            # Edge: Parameter -> Datatype (Links to the type node created in init)
            # Use 'Any' if the type is unknown, to link to the fallback DATATYPE node
            linked_type = param_type.split('[')[0]  # Use base type for simple links (e.g., List[int] -> List)
            self.g.add_edge(param_id, linked_type if linked_type in BUILTIN_TYPES else 'Any', 'is_of_type')

        # 4. Edge: Method -> Return Type
        # Link the method directly to the DATATYPE node it returns
        linked_return_type = return_type.split('[')[0]
        self.g.add_edge(method_id, linked_return_type if linked_return_type in BUILTIN_TYPES else 'Any', 'returns_type')

        self.generic_visit(node)

    # C. Visit Class Variables
    def visit_Assign(self, node: ast.Assign):
        """Identifies class variables and creates CLASS_VAR nodes."""
        if not self.current_class: return

        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Simple type inference (kept simple from original)
                value_type = 'Unknown'
                if isinstance(node.value, ast.Constant):
                    value_type = type(node.value.value).__name__

                var_id = f"{self.current_class}.{var_name}"

                # 1. Add CLASS_VAR Node
                self.g.add_node(var_id, 'CLASS_VAR', {'name': var_name, 'inferred_type': value_type})

                # 2. Edge: Class -> Variable
                self.g.add_edge(self.current_class, var_id, 'has_variable', {'type': value_type})


# --- Step 3: Execution Logic ---

def inspect_code_structure(code_content: str, user_id: str, g_utils_instance: GUtils) -> Dict[str, Any]:
    """
    Parses code content, runs the inspector, and returns the graph admin_data.
    Takes code as a string input, as required by ast.parse.
    """
    try:
        # Check if code is empty
        if not code_content.strip():
            return {"Error": "Input code content is empty."}

        tree = ast.parse(code_content)

        # Initialize and run the inspector
        inspector = StructInspector(g_utils_instance)
        inspector.visit(tree)

        return inspector.g.get_graph_data()

    except Exception as e:
        print(f"❌ Error processing code structure for Benedikt: {e}")
        return {"Error": str(e)}

# Note: You would call `inspect_code_structure` with the code content string
# and an instance of GUtils, like this (example only):
#
# from utils.graph.local_graph_utils import GUtils
#
# code = "class MyClass:\n    x: int = 5\n    def run(self, admin_data: list[int]) -> bool: return True"
# graph_util = GUtils()
# result = inspect_code_structure(code, "my_user", graph_util)
# print(result)
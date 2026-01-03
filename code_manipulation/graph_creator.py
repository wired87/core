import ast

import re
from typing import Union, Optional

from core.app_utils import USER_ID, RUNNABLE_MODULES

from core.module_manager.create_runnable import create_runnable
from utils.graph.local_graph_utils import GUtils


def _get_docstring(node: Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef]) -> str:
    """Extracts docstring from function or class node."""
    return ast.get_docstring(node) or ""


def _get_type_name(node: Optional[ast.expr]) -> str:
    """Extracts type name from annotation or defaults to 'Any'."""
    return ast.unparse(node) if node else 'Any'


class StructInspector(ast.NodeVisitor):

    """
    USE FOR SINGLE FILE
    Traverses AST to populate a CodeGraph with classes, methods, and variables.
    The graph stores the entire structure; no redundant internal dicts are kept.
    """

    def __init__(self, G):
        self.current_class: Optional[str] = None
        self.g = GUtils(G=G)

    # B. Visit Methods (Sync/Async)
    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._process_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._process_function(node)

    def _process_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        """Processes methods (or standalone functions, if not in class)."""
        try:
            #print("node", node, type(node))

            method_name = node.name

            method_id = method_name
            print(f"CREATE METHOD:{self.module_name}:", method_id)

            if not method_id.startswith("_"):

                print("Get Method Data")
                return_type = _get_type_name(node.returns)
                #print("return_type", return_type)

                docstring = _get_docstring(node)
                #print("docstring", docstring)

                return_key = self.extract_return_statement_expression(
                    method_node=node,
                )
                entire_def = ast.unparse(node)

                data = {
                    "nid": method_id,
                    "tid": 0,
                    "parent": ["MODULE"],
                    "type": "METHOD",
                    'return_key': return_key,
                    'returns': return_type,
                    'docstring': docstring,
                    "code": entire_def,
                    "callable": create_runnable(
                        eq_code=entire_def,
                        eq_key=method_id,
                        xtrn_mods=RUNNABLE_MODULES
                    ),
                }

                # 1. Update Parameter Node
                self.g.add_node(
                    attrs=data
                )

                print("METHOD node created", method_id)

                # MODULE -> PARAM
                self.g.add_edge(
                    src=self.module_name,
                    trt=method_id,
                    attrs=dict(
                        rel='has_param',
                        trgt_layer='METHOD',
                        src_layer='MODULE',
                    )
                )

                #print("METHOD-module edge created", method_id)
                self.process_method_params(node, method_id)

        except Exception as e:
            print("Err _process_function", e)

        self.generic_visit(node)


    def process_method_params(self, node, method_id):
        # 3. Process Parameters
        #print("process_method_params node", node, type(method_id))
        for arg in node.args.args:
            try:
                if arg.arg == 'self': continue
                param_name = arg.arg
                param_type = _get_type_name(arg.annotation)

                # Edge: Param -> Parameter (Needs Input)
                self.g.add_edge(
                    src=method_id,
                    trt=param_name,
                    attrs=dict(
                        rel='requires_param',
                        type=param_type,
                        trgt_layer='PARAM',
                        src_layer='METHOD',
                    ))
            except Exception as e:
                print("Err node.args.args", e)


    def extract_return_statement_expression(self, method_node: ast.FunctionDef) -> Optional[str]:
        """
        Extracts the source code representation of the expression being returned.
        This is the programmatic way to get the 'return statement including key'.
        """
        for node in ast.walk(method_node):
            try:
                if isinstance(node, ast.Return):
                    # Check if a value is returned (i.e., not a simple 'return')
                    if node.value is not None:
                        # Use ast.unparse to reconstruct the Python code for the expression
                        # Example: for 'return new_dict', returns 'new_dict'
                        # Example: for 'return laplacian_h', returns 'laplacian_h'
                        return ast.unparse(node.value).strip()
            except Exception as e:
                print("Err extract_return_statement_expression",e)
        return None

    def add_param(self, method_name, body_src):
        for match in re.finditer(r"attrs\[['\"]([^'\"]+)['\"]\]", body_src):
            try:
                attr_key = match.group(1)
                # add attr node
                self.g.add_node(
                    dict(
                        nid=attr_key,
                        type="PARAM",
                        parent=["DATATYPE", "METHOD"],
                        module=self.module_name,
                ))

                # link METHOD -> ATTR
                self.g.add_edge(
                    src=method_name,
                    trt=attr_key,
                    attrs={
                        "rel": "uses_param",
                        "src_layer": "METHOD",
                        "trgt_layer": "PARAM",
                    }
                )

                # MODULE -> PARAM
                self.g.add_edge(
                    src=self.module_name,
                    trt=attr_key,
                    attrs=dict(
                        rel='has_variable',
                        src_layer='MODULE',
                        trgt_layer='PARAM',
                    )
                )
            except Exception as e:
                print("Err add_param",e)

    # C. Visit Class Variables
    def visit_Assign(self, node: ast.Assign):
        """Identifies class variables and creates CLASS_VAR nodes."""

        if not self.current_class: return

        for target in node.targets:
            try:
                if isinstance(target, ast.Name):
                    var_name = target.id

                    # Simple type inference (kept simple from original)
                    value_type = 'Unknown'
                    if isinstance(node.value, ast.Constant):
                        value_tyspe = type(node.value.value).__name__

                    var_id = f"{self.current_class}.{var_name}"

                    # 1. Add CLASS_VAR Node
                    self.g.add_node(
                        dict(
                            nid=var_id,
                            type='CLASS_VAR',
                            name=var_name,
                            inferred_type=value_type,
                        )
                    )
            except Exception as e:
                print("Err Agign ", e)

    def convert_module_to_graph(self, code_content, module_name):
        """
        Parses code content, runs the inspector, and returns the graph admin_data.
        Takes code as a string input, as required by ast.parse.
        """
        self.module_name = module_name
        print("ADD MODULE:", module_name)
        try:
            # Check if code is empty
            if not code_content.strip():
                return {
                    "Error": "Input code content is empty."
                }

            tree = ast.parse(code_content)
            self.visit(tree)

        except Exception as e:
            print(f"❌ Error processing code structure for {module_name}: {e}")




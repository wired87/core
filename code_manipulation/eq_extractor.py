"""
Takes an str parsed py def:
recognize an eq ->

"""
import ast
from typing import Set

BUILTIN_TYPES: Set[str] = {
    'str', 'int', 'float', 'bool', 'list','tuple', 'Any',
    'List', 'Tuple', 'array',
}


class EqExtractor:

    def __init__(self, g):
        self.g = g
        self.init_data_type=False

    def process_equation(
        self,
        expression_node,
        parent_id,
        module_id,
    ):
        # XTRACT METHODS EQ AND LINK PARAMS -> OPERATOR -> ETC
        if self.init_data_type is False:
            self.init_data_type_nodes()

        self.analyze_expression_and_add_edges(
            expression_node,
            parent_id,
            module_id,
        )

        print("EqExtractor process finished")


    def analyze_expression_and_add_edges(
            self,
            expression_node,
            parent_id,
            module_id,
    ):
        """
        Rekursiver AST-Visitor zur Erkennung von Operatoren und Parametern.
        Gibt den Namen des finalen Ergebnisknotens zurück.
        """
        if isinstance(expression_node, ast.BinOp):
            # 1. Erstelle einen Operator-Knoten für die Operation (z.B. '+', '*')
            op_symbol = type(expression_node.op).__name__.lower()
            operator_node_id = f"{parent_id}_{op_symbol}_{hash(expression_node)}"

            self.g.add_node(attrs={
                "nid": operator_node_id,
                "type": "OPERATOR",
                "op": op_symbol
            })

            # 2. Verbinde den vorherigen Schritt (parent_id) mit diesem Operator
            self.g.add_edge(src=parent_id, trgt=operator_node_id, rel='output_of')

            # 3. Rekursiv die linken und rechten Operanden verarbeiten (Klammer-Handling)
            # Die Struktur der BinOp (left, op, right) löst Klammern wie a*(b+c) natürlich auf
            left_input_id = self.analyze_expression_and_add_edges(
                expression_node.left, operator_node_id, module_id
            )
            right_input_id = self.analyze_expression_and_add_edges(
                expression_node.right, operator_node_id, module_id
            )

            # Füge Kanten vom Operator zu seinen Inputs hinzu
            self.g.add_edge(src=operator_node_id, trgt=left_input_id, rel='input_a')
            self.g.add_edge(src=operator_node_id, trgt=right_input_id, rel='input_b')

            return operator_node_id  # Das Ergebnis dieser Operation

        elif isinstance(expression_node, ast.Call):
            # 1. Handling für Funktionen wie np.sum, jnp.dot (AST.Call)
            # Identifiziere den Funktionsnamen (z.B. 'sum', 'dot')
            func_name = ast.unparse(expression_node.func)
            call_node_id = f"{parent_id}_call_{func_name}_{hash(expression_node)}"

            self.g.add_node(attrs={"nid": call_node_id, "type": "FUNCTION_CALL", "func": func_name})
            self.g.add_edge(src=parent_id, trgt=call_node_id, rel='output_of')

            # 2. Verarbeite alle Argumente der Funktion rekursiv
            for i, arg in enumerate(expression_node.args):
                arg_input_id = self.analyze_expression_and_add_edges(arg, call_node_id, module_id)
                self.g.add_edge(src=call_node_id, trgt=arg_input_id, rel=f'arg_{i}')

            return call_node_id

        elif isinstance(expression_node, ast.Name):
            # 1. Handling für Variablen (z.B. 'psi', 'h')
            param_name = expression_node.id

            # Stellen Sie sicher, dass die Variable als Parameter-Knoten existiert
            if not self.g.has_node(param_name):
                self.g.add_node(
                    attrs={"nid": param_name, "type": "PARAM"})

            # Füge eine Kante von der Funktion zur benötigten Variable hinzu
            self.g.add_edge(src=module_id, trgt=param_name, rel='requires_param_in_body')

            return param_name

        else:
            # Handling von Literalen (Zahlen) und anderen Knoten
            return str(ast.unparse(expression_node))  # Rückgabe des Literalwertes



    def init_data_type_nodes(self):
        """Creates nodes for all predefined built-in and array types."""
        try:
            for data_type in BUILTIN_TYPES:
                # Node ID is the type name itself
                self.g.add_node(
                    dict(
                        nid=data_type,
                        parent=['DATATYPE'],
                        type=data_type,
                    )
                )
        except Exception as e:
            print(f"Err init_data_type_nodes: {e}")
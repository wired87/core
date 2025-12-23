import re
import sympy as sp
from typing import List, Dict, Any, Optional

class ParamOperatorConnector:
    """
    Ray remote constructor node that connects PARAMS nodes to OPERATOR nodes
    based on equation parameter matching.
    
    Workflow:
    1. Receives graph with PARAMS and OPERATOR nodes
    2. Receives list of equations
    3. For each equation, extracts parameters
    4. For each PARAM node, checks if it matches any parameter in any equation
    5. If match found, adds edge from PARAM node to OPERATOR node
    """

    def __init__(self, equations: List[str], utils_worker_name: str = "UTILS_WORKER"):
        """
        Initialize the connector.
        
        Args:
            equations: List of equation strings to analyze
            utils_worker_name: Name of the UTILS_WORKER actor to access the graph
        """
        #BaseActor.__init__(self)
        self.equations = equations
        self.utils_worker_name = utils_worker_name
        print(f"ParamOperatorConnector initialized with {len(equations)} equations")

    def extract_parameters_from_equation(self, equation: str) -> List[str]:
        """
        Extract parameter names from an equation string.
        
        Uses sympy to parse the equation and extract free symbols.
        Falls back to regex if sympy parsing fails.
        
        Args:
            equation: Equation string (e.g., "x + y * z", "f(a, b, c)")
            
        Returns:
            List of parameter names found in the equation
        """
        params = []
        
        try:
            # Try using sympy to parse the equation
            # Remove common function names and operators that aren't variables
            expr = sp.sympify(equation, evaluate=False)
            free_symbols = expr.free_symbols
            params = [str(symbol) for symbol in free_symbols]
        except Exception:
            # Fallback to regex extraction
            # Match valid Python identifiers (variable names)
            # Exclude common operators and built-in functions
            pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
            matches = re.findall(pattern, equation)
            
            # Filter out common operators and built-in functions
            excluded = {
                'and', 'or', 'not', 'in', 'is', 'if', 'else', 'for', 'while',
                'def', 'class', 'import', 'from', 'return', 'True', 'False',
                'None', 'abs', 'min', 'max', 'sum', 'len', 'range', 'print',
                'sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'pi', 'e'
            }
            
            params = [m for m in matches if m not in excluded]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_params = []
        for p in params:
            if p not in seen:
                seen.add(p)
                unique_params.append(p)
        
        return unique_params

    def _get_utils_worker(self):
        """Get the UTILS_WORKER actor reference."""
        return get_actor(name=self.utils_worker_name)
    
    def _get_graph(self):
        """Get the graph from UTILS_WORKER."""
        utils_worker = self._get_utils_worker()
        return ray.get(utils_worker.get_data_state_G.remote())
    
    def get_nodes_by_type(self, node_types: List[str]) -> Dict[str, List[tuple]]:
        """
        Get all nodes from the graph filtered by type.
        
        Args:
            node_types: List of node types to filter (e.g., ["PARAMS", "OPERATOR"])
            
        Returns:
            Dictionary mapping node type to list of (node_id, node_attrs) tuples
        """
        G = self._get_graph()
        nodes_by_type = {ntype: [] for ntype in node_types}
        
        for nid, attrs in G.nodes(data=True):
            ntype = attrs.get("type", "").upper()
            if ntype in [t.upper() for t in node_types]:
                nodes_by_type[ntype].append((nid, attrs))
        
        return nodes_by_type

    def connect_params_to_operators(self) -> Dict[str, Any]:
        """
        Main method to connect PARAMS nodes to OPERATOR nodes based on equations.
        
        Workflow:
        1. Get all PARAMS and OPERATOR nodes from graph
        2. Extract parameters from each equation
        3. For each PARAM node, check if its name matches any parameter in any equation
        4. If match found, add edge from PARAM node to corresponding OPERATOR node
        
        Returns:
            Dictionary with statistics about connections made
        """
        # Get nodes by type
        nodes_by_type = self.get_nodes_by_type(["PARAMS", "OPERATOR"])
        param_nodes = nodes_by_type.get("PARAMS", [])
        operator_nodes = nodes_by_type.get("OPERATOR", [])
        
        print(f"Found {len(param_nodes)} PARAMS nodes and {len(operator_nodes)} OPERATOR nodes")
        
        # Extract parameters from each equation
        # Map equation index to its parameters
        equation_params = {}
        for idx, equation in enumerate(self.equations):
            params = self.extract_parameters_from_equation(equation)
            equation_params[idx] = params
            print(f"Equation {idx}: {equation} -> Parameters: {params}")
        
        # Create a mapping from parameter name to operator node IDs
        # Assuming each equation corresponds to an OPERATOR node
        # If there are more equations than operators, we'll match by index
        # If there are more operators than equations, we'll only process matching indices
        
        edges_created = 0
        edges_skipped = 0
        
        # For each PARAM node, check if it matches any parameter in any equation
        for param_nid, param_attrs in param_nodes:
            param_name = param_attrs.get("nid", param_nid)  # Use nid as param name if no explicit name
            
            # Also check for common name fields
            if "name" in param_attrs:
                param_name = param_attrs["name"]
            elif "param_name" in param_attrs:
                param_name = param_attrs["param_name"]
            
            param_name = str(param_name).lower()  # Normalize to lowercase for matching
            
            # Check each equation for this parameter
            for eq_idx, eq_params in equation_params.items():
                # Normalize equation parameters to lowercase for matching
                eq_params_lower = [p.lower() for p in eq_params]
                
                if param_name in eq_params_lower:
                    # Found a match! Now find the corresponding OPERATOR node
                    # Try to match by index first
                    if eq_idx < len(operator_nodes):
                        operator_nid, operator_attrs = operator_nodes[eq_idx]
                        
                        # Get graph to check if edge exists
                        G = self._get_graph()
                        if not G.has_edge(param_nid, operator_nid):
                            # Add edge from PARAM to OPERATOR via UTILS_WORKER
                            utils_worker = self._get_utils_worker()
                            ray.get(utils_worker.add_edge.remote(
                                src=param_nid,
                                trgt=operator_nid,
                                attrs=dict(
                                    rel="uses_param",
                                    src_layer="PARAMS",
                                    trgt_layer="OPERATOR",
                                    equation_idx=eq_idx,
                                    param_name=param_name
                                )
                            ))
                            edges_created += 1
                            print(f"Created edge: {param_nid} -> {operator_nid} (equation {eq_idx}, param: {param_name})")
                        else:
                            edges_skipped += 1
                            print(f"Edge already exists: {param_nid} -> {operator_nid}")
                    else:
                        # If no operator node at this index, try to find by name or other attributes
                        # For now, we'll skip if index doesn't match
                        print(f"Warning: Equation index {eq_idx} exceeds operator nodes count")
        
        result = {
            "edges_created": edges_created,
            "edges_skipped": edges_skipped,
            "param_nodes_count": len(param_nodes),
            "operator_nodes_count": len(operator_nodes),
            "equations_count": len(self.equations)
        }
        
        print(f"Connection complete: {edges_created} edges created, {edges_skipped} skipped")
        return result

    def run(self) -> Dict[str, Any]:
        """
        Execute the connection process.
        
        Returns:
            Dictionary with connection statistics
        """
        return self.connect_params_to_operators()


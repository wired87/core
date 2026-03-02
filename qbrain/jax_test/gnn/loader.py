import json
import jax
import jax.numpy as jnp
from jax import jit, vmap
from flax import nnx
from typing import List, Tuple, Any

from mod import Node

def recursive_tuple_conversion(item):
    """Recursively converts lists to tuples."""
    if isinstance(item, list):
        return tuple(recursive_tuple_conversion(x) for x in item)
    return item

def compile_function(func_str: str, method_id: str):
    """
    Compiles a function string into a callable.
    CAUTION: Uses exec(). Ensure input is trusted.
    """
    # specific context with necessary imports
    local_scope = {}
    global_scope = {
        'jax': jax,
        'jnp': jnp,
        'jit': jit,
        'vmap': vmap,
        'Tuple': Tuple,
        'List': List,
        # Add any other required globals here
    }
    
    try:
        exec(func_str, global_scope, local_scope)
    except Exception as e:
        raise ValueError(f"Failed to compile function for method {method_id}: {e}")
    
    # We assume the function name is extracted or known. 
    # Based on the JSON snippet, the function name is likely the first defined function.
    # Let's find the first callable in local_scope.
    func = None
    for k, v in local_scope.items():
        if callable(v):
            func = v
            break # Assume the first one is the main one
            
    if func is None:
        raise ValueError(f"No callable found in code string for method {method_id}")
        
    return func

def load_chains_from_json(json_path: str, rngs: nnx.Rngs) -> List[Node]:
    """
    Parses the JSON file and constructs a list of Node objects (chains).
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    updater_patterns = data.get("UPDATOR_PATTERN", [])
    chains = []
    
    for idx, item in enumerate(updater_patterns):
        # Expected structure based on snippet: 
        # [code_string, method_id_int, null, input_pattern, output_pattern]
        # or similar. Let's adapt robustly.
        
        if len(item) < 5:
            print(f"Skipping item {idx}: unexpected length {len(item)}")
            continue
            
        code_str = item[0]
        method_id_raw = item[1] # int likely
        # item[2] appears to be None/null
        inp_pattern_raw = item[3]
        outp_pattern_raw = item[4]
        
        # safely recursive convert patterns to tuples
        inp_pattern = recursive_tuple_conversion(inp_pattern_raw)
        outp_pattern = recursive_tuple_conversion(outp_pattern_raw)
        
        # The structure of inp_pattern in JSON seems to be nested lists.
        # Node expects inp_patterns: List[Tuple] -> list of (method, pattern...)
        # In mod.py: inp_patterns: List[Tuple]
        # Verify if inp_pattern needs to be a list of tuples or just a tuple.
        # JSON snippet shows: [[[[[0, 2, 0], ...]]]]
        # which converts to tuple of tuples.
        
        compiled_func = compile_function(code_str, str(method_id_raw))
        
        # Define in_axes - usually inferred or standard. 
        # mod.py Node takes in_axes_def. 
        # For now, we might default it or try to parse it if available.
        # The JSON doesn't seem to explicitly have in_axes. 
        # We will assume standard structure or use checking.
        # For this implementation, we default to mapping over the first argument (0,).
        
        chain = Node(
            runnable=compiled_func,
            inp_patterns=list(inp_pattern) if isinstance(inp_pattern, (list, tuple)) else [inp_pattern], 
            outp_pattern=outp_pattern,
            in_axes_def=(0,), # Default assumption, user might need to refine
            method_id=str(method_id_raw),
            rngs=rngs
        )
        chains.append(chain)
        
    print(f"Loaded {len(chains)} chains from {json_path}")
    return chains

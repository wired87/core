"""
Generated model from guard components. Load with: from model_output import load_components
"""
import json
import os

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_output.json")

def load_components():
    """Load the components dict from model_output.json."""
    with open(_MODEL_PATH, "r") as f:
        return json.load(f)

COMPONENTS = load_components()

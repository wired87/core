# The LLM-generated code string
class DataProcessor:pass

class_code_string = """
class DataProcessor:
    # Class initializer (constructor)
    def __init__(self, multiplier):
        # Store the multiplier as an instance variable
        self.multiplier = multiplier

    # Class method to perform the operation
    def process(self, value):
        # Use the stored multiplier
        return value * self.multiplier
"""

def get_class_from_code_str(code_str):
    exec(code_str, globals())
    return globals().get("DataProcessor")

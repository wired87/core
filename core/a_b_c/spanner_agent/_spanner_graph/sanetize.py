import re

def sanitize_for_spanner(input_string):
    """
    Sanitizes a string for Google Cloud Spanner, allowing specific characters.

    Args:
        input_string: The string to sanitize.

    Returns:
        The sanitized string.
    """
    #print("Value before", input_string)
    if not isinstance(input_string, str):
        return str(input_string)  # Handle ints, floats, etc.

    # Allow alphanumeric, underscore, period, dash, space, colon, slash, exclamation, and question mark.
    allowed_chars = r"a-zA-Z0-9_.\- :/\!\?"
    sanitized_string = re.sub(f"[^{allowed_chars}]", "", input_string)

    # Replace single quotes, which are a common offender.
    sanitized_string = sanitized_string.replace("'", "")  # Or replace with a different character if needed.

    # Remove multiple spaces
    sanitized_string = re.sub(r"\s+", " ", sanitized_string).strip()

    #print("Sanitized value", sanitized_string)
    return sanitized_string